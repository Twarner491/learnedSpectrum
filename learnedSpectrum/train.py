import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import ViTModel, ViTConfig
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from einops import rearrange, repeat
from typing import Optional, Tuple
from functools import partial
from tqdm import tqdm
import random
import os
import math


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


from learnedSpectrum.config import Config, DataConfig
from learnedSpectrum.data import DatasetManager, create_dataloaders
from learnedSpectrum.utils import (
    seed_everything, get_optimizer, verify_model_devices,
    calculate_metrics, save_checkpoint, load_checkpoint, print_gpu_memory
)


logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """causal self-attention w/ hemodynamic mask"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # hemodynamic response mask - 6s BOLD delay
        mask = torch.triu(torch.ones(64, 64), diagonal=3)
        self.register_buffer('mask', mask.float().masked_fill(mask == 0, float('-inf')))
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:N, :N] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        # reduced chunk size for stability
        max_seq = 128
        if x.size(1) > max_seq:
            chunks = []
            for i in range(0, x.size(1), max_seq):
                chunk = x[:, i:i+max_seq]
                chunk = chunk + self.attn(
                    *[self.norm1(chunk)]*3,
                    need_weights=False
                )[0]
                chunk = chunk + self.mlp(self.norm2(chunk))
                chunks.append(chunk)
            return torch.cat(chunks, dim=1)
            
        x = x + self.attn(*[self.norm1(x)]*3, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))
    

class VisionTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1_lambda = 0  # was missing
        
        # nets 
        self.temporal_net = nn.Sequential(
            nn.Conv3d(8, 16, 1),  
            nn.BatchNorm3d(16),
            nn.GELU(),  # smoother + better bp
            nn.AvgPool3d((1,1,4))
        )
        
        self.patch_embed = nn.Conv2d(
            16, config.EMBED_DIM//2, 
            kernel_size=config.PATCH_SIZE, 
            stride=config.PATCH_SIZE//2  # overlap
        )
        
        # structural params
        self.chunk_size = 16
        self.sparse_samples = 4
        self.pool_factor = 16
        self.subsample_rate = 2
        self.max_seq_len = 2048
        
        # pos embed + cls
        num_patches = ((64 // (config.PATCH_SIZE//2)) ** 2) + 1
        self.register_buffer('pos_embed', self._get_sincos_pos_embed(
            config.EMBED_DIM//2, num_patches))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.EMBED_DIM//2))
        
        # transformer w/ progressive dropout
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.EMBED_DIM//2, 
                config.NUM_HEADS//2,
                mlp_ratio=2.0,
                drop=config.DROP_RATE * (i+1)/config.DEPTH
            ) for i in range(config.DEPTH//2)
        ])
        
        # classifier
        self.head = nn.Sequential(
            nn.LayerNorm(config.EMBED_DIM//2),
            nn.Linear(config.EMBED_DIM//2, config.NUM_CLASSES)
        )
        
        self._init_weights()
        self.to(config.device)
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight)
        
    def _get_sincos_pos_embed(self, embed_dim, num_patches):
        pos = torch.arange(num_patches).unsqueeze(1).float()
        omega = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            -(math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = torch.sin(pos * omega.T)
        pe[:, 1::2] = torch.cos(pos * omega.T)
        return pe.unsqueeze(0)

    def _pad_temporal(self, x, target_len):
        curr_len = x.size(-1)
        if curr_len >= target_len:
            return x
        pad_len = target_len - curr_len
        last_slice = x[..., -1:]
        padding = last_slice.expand(*x.shape[:-1], pad_len)
        return torch.cat([x, padding], dim=-1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        B = x.shape[0]
        outputs = []
        
        # temporal norm
        if x.size(-1) > self.max_seq_len:
            start_idx = torch.randint(0, x.size(-1) - self.max_seq_len, (1,))
            x = x[..., start_idx:start_idx + self.max_seq_len]
        else:
            x = self._pad_temporal(x, self.max_seq_len)
        
        x = x[..., ::self.subsample_rate]
        
        # chunk processing
        for t_start in range(0, x.size(-1), self.chunk_size):
            t_end = min(t_start + self.chunk_size, x.size(-1))
            chunk = x[..., t_start:t_end].permute(0, 3, 1, 2, 4)
            t_indices = torch.randperm(chunk.size(-1))[:self.sparse_samples]
            chunk = chunk[..., t_indices]
            
            # temporal conv
            convd = []
            for t in range(0, chunk.size(-1), self.pool_factor):
                t_slice = chunk[..., t:t+self.pool_factor].mean(dim=-1)
                t_out = self.temporal_net(t_slice.unsqueeze(2))
                spatial = t_out.squeeze(2)
                
                # patch + pos
                patch = self.patch_embed(spatial).flatten(2).transpose(1, 2)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                patch = torch.cat([cls_tokens, patch], dim=1)
                patch = patch + self.pos_embed[:, :patch.size(1)]
                
                # transformer pass
                for block in self.blocks:
                    patch = block(patch)
                
                convd.append(patch[:, 0])
            
            outputs.append(torch.stack(convd).mean(0))
        
        return self.head(torch.stack(outputs).mean(0))

    def l1_reg(self):
        return self.l1_lambda * sum(p.abs().sum() for p in self.parameters() if p.dim() > 1)
    

def cutmix(x, y, alpha=1.0):
    """cutmix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """helper for cutmix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(model, loader, optimizer, scheduler, scaler, config, pbar=None):
    model.train()
    total_loss = 0
    accum_steps = max(1, 32 // config.BATCH_SIZE)
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(config.device, dtype=torch.float16)
        labels = labels.to(config.device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs) 
            loss = F.cross_entropy(outputs, labels) + model.l1_reg()
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
            
            if pbar:
                pbar.set_postfix({'train_loss': f'{loss.item():.3f}'}, refresh=True)
        
        total_loss += loss.item() * accum_steps
    
    return total_loss / len(loader)


def train_loop(model, train_dl, val_dl, opt, sched, scaler, config):
    best_loss = float('inf')
    total_steps = config.NUM_EPOCHS * len(train_dl)
    current_step = 0
    
    # single progress bar w/ fixed width
    pbar = tqdm(total=total_steps, 
                desc='training',
                ncols=min(100, get_terminal_size().columns//2),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss = 0
        
        for i, (inputs, labels) in enumerate(train_dl):
            inputs = inputs.to(config.device, dtype=torch.float16)
            labels = labels.to(config.device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels) + model.l1_reg()
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
            if sched: sched.step()
            
            train_loss += loss.item()
            current_step += 1
            
            # update progress every few steps
            if current_step % 5 == 0:
                avg_loss = train_loss / (i + 1)
                pbar.set_description(f'e{epoch:02d} loss:{avg_loss:.3f}')
                pbar.update(5)
        
        # validation
        val_loss, val_metrics = evaluate(model, val_dl, config)
        pbar.set_postfix({'val_loss': f'{val_loss:.3f}', 
                         'val_acc': f'{val_metrics["accuracy"]:.3f}'}, 
                         refresh=True)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, opt, epoch, val_loss, config, 
                          f"best_model_epoch_{epoch}.pth")
    
    pbar.close()
    return model


def evaluate(model, loader, config):
    """eval w/ metrics aligned to visualization expectations"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            all_preds.append(outputs)
            all_labels.append(labels)
    
    # consolidate predictions
    outputs = torch.cat(all_preds)
    targets = torch.cat(all_labels)
    
    # calc metrics w/ existing util
    metrics = calculate_metrics(outputs, targets)
    
    return total_loss / len(loader), metrics


def main():
    config = Config()
    data_config = DataConfig()
    Path(config.CKPT_DIR).mkdir(exist_ok=True)

    # load data
    dm = DatasetManager(config, data_config)
    train_ds, val_ds, test_ds = dm.prepare_datasets()
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config)

    # init model w/ aggressive optimization
    model = VisionTransformerModel(config).half().to(config.device)
    opt = get_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')

    print('\nstarting training...\n')  # force tqdm to new line
    epochs = range(config.NUM_EPOCHS)
    pbar = tqdm(epochs, bar_format='{l_bar}{bar:30}{r_bar}', ncols=80)
    
    for epoch in pbar:
        # train
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = F.cross_entropy(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
            losses.append(loss.item())
            if i % 5 == 0:  # faster updates
                pbar.set_description(f'e{epoch} l:{np.mean(losses):.3f}')

        # eval 
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.inference_mode():
            for x, y in val_dl:
                x, y = x.to(config.device), y.to(config.device)
                out = model(x)
                val_loss += F.cross_entropy(out, y).item()
                correct += (out.argmax(-1) == y).sum().item()
                total += y.size(0)
        
        val_loss /= len(val_dl)
        acc = correct / total
        
        # save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Path(config.CKPT_DIR) / "best.pt")
        
        pbar.set_postfix_str(f'val_l:{val_loss:.3f} acc:{acc:.3f}')
        

if __name__ == "__main__":
    main()
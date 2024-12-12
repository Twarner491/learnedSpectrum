import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import ViTModel, ViTConfig
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score, 
    recall_score, 
    f1_score
)
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
        self.norm2 = nn.LayerNorm(dim)
        
        # Scaled residual parameters
        self.gamma1 = nn.Parameter(torch.ones(dim) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 0.1)
        
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop)  # Additional dropout after projection
        )
        
    def forward(self, x):
        # Pre-norm architecture
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.gamma1.unsqueeze(0).unsqueeze(0) * attn_out
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.gamma2.unsqueeze(0).unsqueeze(0) * mlp_out
        return x
    

class VisionTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1_lambda = 1e-5
        self.dropout = nn.Dropout(0.2)
        self.l2_lambda = 1e-4
        
        # Model parameters
        self.subsample_rate = 4
        self.chunk_size = 4
        self.sparse_samples = 1
        self.pool_factor = 64
        self.max_seq_len = 1024
        
        # Channel reduction first
        self.channel_reduce = nn.Sequential(
            nn.Conv3d(30, 16, 1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 16, 1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Dropout3d(0.1)
        )
        
        # Then temporal network with residual connections
        self.temporal_net = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            ResidualBlock3d(16, 16),  # Keep same channels for residual
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Dropout3d(0.1)
        )
        
        # Rest of the initialization remains the same
        self.input_norm = nn.LayerNorm([16, 64, 64])
        
        self.patch_embed = nn.Conv2d(
            16, config.EMBED_DIM//2,
            kernel_size=config.PATCH_SIZE,
            stride=config.PATCH_SIZE//2,
            bias=False
        )
        
        # structural params
        self.max_seq_len = 1024
        
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
            nn.Dropout(0.2),
            nn.Linear(config.EMBED_DIM//2, config.NUM_CLASSES)
        )
        
        self._init_weights()
        
        # Move to device first
        self.to(config.device)
        
        # Keep BatchNorm in FP32
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.float()
        
    def _init_weights(self):
        """Improved weight initialization"""
        # Initialize channel reduction
        for m in self.channel_reduce.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize temporal network
        for m in self.temporal_net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize embedding layers
        nn.init.normal_(self.cls_token, std=0.01)
        nn.init.normal_(self.patch_embed.weight, std=0.01)
        
        # Initialize transformer blocks
        for block in self.blocks:
            # Initialize MultiheadAttention weights
            nn.init.normal_(block.attn.in_proj_weight, std=0.01)
            nn.init.normal_(block.attn.out_proj.weight, std=0.01)
            
            if block.attn.in_proj_bias is not None:
                nn.init.zeros_(block.attn.in_proj_bias)
            if block.attn.out_proj.bias is not None:
                nn.init.zeros_(block.attn.out_proj.bias)
            
            # Initialize MLP weights - find Linear layers specifically
            for module in block.mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Initialize classification head
        if hasattr(self, 'head'):
            nn.init.zeros_(self.head[-1].bias)
            nn.init.normal_(self.head[-1].weight, std=0.02)

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

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Basic temporal subsampling
        x = x[..., ::self.subsample_rate]
        
        # 2. Channel reduction and temporal processing
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = self.channel_reduce(x)
        x = self.temporal_net(x)
        
        # 3. Reshape for transformer
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)
        
        # 4. Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 5. Add position embeddings and CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        
        # 6. Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 7. Reshape and pool temporal dimension
        x = x.view(B, T, -1, x.size(-1))  # [B, T, N, D]
        x = x.mean(dim=1)  # Average pool temporal
        
        # 8. Classification
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        
        return x

    def l1_reg(self):
        return self.l1_lambda * sum(p.abs().sum() for p in self.parameters() if p.dim() > 1)
    
    def l2_reg(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return self.l2_lambda * l2_loss
    
    def get_attention_weights(self, x):
        torch.cuda.empty_cache()
        B = x.shape[0]
        attn_weights = []
        
        x = x.to(dtype=torch.float16)
        
        if x.size(-1) > self.max_seq_len:
            x = x[..., :self.max_seq_len]  # Deterministic truncation
        else:
            x = self._pad_temporal(x, self.max_seq_len)
        
        x = x[..., ::self.subsample_rate]
        
        for t_start in range(0, x.size(-1), self.chunk_size):
            torch.cuda.empty_cache()
            
            t_end = min(t_start + self.chunk_size, x.size(-1))
            chunk = x[..., t_start:t_end].permute(0, 3, 1, 2, 4)
            t_indices = torch.arange(0, chunk.size(-1), 2)[:self.sparse_samples]
            chunk = chunk[..., t_indices]
            
            for t in range(0, chunk.size(-1), self.pool_factor):
                with torch.cuda.amp.autocast():
                    t_slice = chunk[..., t:t+self.pool_factor].mean(dim=-1)
                    t_out = self.temporal_net(t_slice.unsqueeze(2))
                    spatial = t_out.squeeze(2)
                    
                    patch = self.patch_embed(spatial).flatten(2).transpose(1, 2)
                    cls_tokens = self.cls_token.expand(B, -1, -1)
                    patch = torch.cat([cls_tokens, patch], dim=1)
                    patch = patch + self.pos_embed[:, :patch.size(1)]
                    
                    for block in self.blocks:
                        with torch.no_grad():
                            _, attn = block.attn(
                                *[block.norm1(patch)]*3,
                                need_weights=True
                            )
                            attn_weights.append(attn.cpu())  # Move to CPU immediately
                            
        torch.cuda.empty_cache()
        
        return torch.stack(attn_weights)


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


def get_grad_norm(model):
    """Calculate gradient norm across all model parameters"""
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_one_epoch(model, loader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        
        x = x.to(config.device, non_blocking=True)
        y = y.to(config.device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):  # Fixed deprecated warning
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        
        if not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        if i % 10 == 0:  # Print progress more frequently
            print(f"Batch {i}/{len(loader)}, Loss: {loss.item():.4f}, "
                  f"Acc: {correct/total:.4f}")
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total
    }


def train_loop(model, train_dl, val_dl, config):
    # Validation set size check and potential fix
    num_val_samples = len(val_dl.dataset)
    min_val_samples = 100
    
    if num_val_samples < min_val_samples:
        logger.warning(
            f"Validation set too small ({num_val_samples} < {min_val_samples}). "
            "Using stratified K-fold validation instead."
        )
        # Implement k-fold validation here
        
    # Initialize training components
    criterion = LabelSmoothingLoss(classes=config.NUM_CLASSES, smoothing=0.1)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase with gradient accumulation
        model.train()
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, 
            scaler, config, max_grad_norm
        )
        
        # Validation phase
        val_loss, val_metrics = evaluate(model, val_dl, config)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
            
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, opt, epoch, val_loss, config)
    
    return model


def evaluate(model, loader, config):
    """Improved evaluation with detailed metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for inputs, labels in loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get unique classes actually present in the data
    unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': (all_preds == all_labels).mean(),
        'class_accuracies': {
            i: (all_preds[all_labels == i] == i).mean() 
            for i in range(config.NUM_CLASSES)
            if i in unique_classes
        },
        'confusion': confusion_matrix(
            all_labels, all_preds, 
            labels=range(config.NUM_CLASSES)
        ),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds)
    }
    
    return metrics['loss'], metrics


def load_best_model(model, config):
    best_model_path = Path(config.CKPT_DIR) / "best_model.pth"
    
    # Try loading best_model.pth first
    try:
        model, _, _ = load_checkpoint(model, None, best_model_path)
        logger.info(f"Loaded best model from {best_model_path}")
        return model
    except FileNotFoundError:
        # If not found, try to find the latest epoch checkpoint
        checkpoints = list(Path(config.CKPT_DIR).glob("best_model_epoch_*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
            model, _, _ = load_checkpoint(model, None, latest)
            logger.info(f"Loaded latest checkpoint from {latest}")
            return model
        else:
            logger.warning("No checkpoints found, using fresh model")
            return model


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelWithRegularization(nn.Module):
    def __init__(self, base_model, weight_decay=1e-5):
        super().__init__()
        self.base_model = base_model
        self.weight_decay = weight_decay
    
    def forward(self, x):
        return self.base_model(x)
    
    def l1_reg(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.weight_decay * l1_loss


def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps, min_lr=1e-6):
    """Cosine schedule with warmup and minimum learning rate"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=9, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.GELU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


def perform_final_evaluation(model, test_loader, config, viz, device, class_names):
    """
    Perform comprehensive final evaluation of the model.
    """
    test_loss, test_metrics = evaluate(model, test_loader, config)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        per_class_metrics[class_name] = {
            'precision': precision_score(all_labels == i, all_preds == i),
            'recall': recall_score(all_labels == i, all_preds == i),
            'f1': f1_score(all_labels == i, all_preds == i),
            'support': np.sum(class_mask)
        }
    
    # Create visualizations
    viz.create_final_evaluation_plots(
        all_labels=all_labels,
        all_preds=all_preds,
        all_probs=all_probs,
        class_names=class_names,
        model=model
    )
    
    return test_loss, test_metrics, per_class_metrics


def main():
    config = Config()
    data_config = DataConfig()
    Path(config.CKPT_DIR).mkdir(exist_ok=True)

    # Load data
    dm = DatasetManager(config, data_config)
    train_ds, val_ds, test_ds = dm.prepare_datasets()
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config)

    # Initialize model and training components
    model = VisionTransformerModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Single scheduler initialization
    num_training_steps = len(train_dl) * config.NUM_EPOCHS
    num_warmup_steps = len(train_dl) * 3  # 3 epochs warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr=1e-6
    )

    criterion = LabelSmoothingLoss(classes=config.NUM_CLASSES, smoothing=0.1)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, config
        )
        
        val_loss, val_metrics = evaluate(model, val_dl, config)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config)


if __name__ == "__main__":
    main()
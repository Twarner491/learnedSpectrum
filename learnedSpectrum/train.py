"""
multi-dataset fmri stage clf. handles temporal + spatial heterogeneity.  
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from learnedSpectrum.config import Config, DataConfig
from learnedSpectrum.data import DatasetManager, create_dataloaders
from learnedSpectrum.utils import (
    seed_everything, get_optimizer, verify_model_devices,
    calculate_metrics, save_checkpoint, load_checkpoint
)

logger = logging.getLogger(__name__)

class VisionTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_drop = nn.Dropout2d(0.1)
        self.site_norm = nn.GroupNorm(6, config.VOLUME_SIZE[2])
        self.l1_lambda = 1e-4  # crucial
        
        vit_config = ViTConfig(
            image_size=64,
            patch_size=config.PATCH_SIZE,
            num_channels=config.VOLUME_SIZE[2],
            hidden_size=config.EMBED_DIM,
            num_hidden_layers=config.DEPTH,
            num_attention_heads=config.NUM_HEADS,
            intermediate_size=int(config.EMBED_DIM * config.MLP_RATIO),
            hidden_dropout_prob=config.DROP_RATE,
            attention_probs_dropout_prob=config.ATTN_DROP_RATE,
            output_attentions=True
        )
        self.vit = ViTModel(vit_config)
        self.classifiers = nn.ModuleList([
            nn.utils.spectral_norm(nn.Linear(config.EMBED_DIM, config.NUM_CLASSES))
            for _ in range(3)
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, return_attn=False):
        x = x.to(self.device).squeeze(-1).permute(0, 3, 1, 2)
        x = self.site_norm(x)
        
        if x.shape[-1] != 64 or x.shape[-2] != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            
        x = self.spatial_drop(x)
        if self.training:
            x = x + torch.randn_like(x) * 0.01
            
        outputs = self.vit(pixel_values=x, output_attentions=return_attn)
        logits = torch.stack([clf(outputs.last_hidden_state[:, 0]) for clf in self.classifiers])
        
        if return_attn:
            return logits.mean(0), outputs.attentions[-1]
        return logits.mean(0)
        
    def l1_reg(self):
        """l1 on non-bn params"""
        return self.l1_lambda * sum(p.abs().sum() for p in self.parameters() if len(p.shape) > 1)

def train_one_epoch(model, loader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0.0
    
    for i, (inputs, labels) in enumerate(loader):
        batch_loss = 0
        optimizer.zero_grad()
        
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        
        # 2x aug per batch  
        for _ in range(2):
            if np.random.random() < 0.5:
                idx = torch.randperm(inputs.size(0), device=inputs.device)
                lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                mixed_x = lam * inputs + (1 - lam) * inputs[idx]
                mixed_y = labels.clone()
                mixed_y[idx] = labels[idx]
            else:
                mixed_x = inputs
                mixed_y = labels
            
            outputs = model(mixed_x)
            loss = nn.CrossEntropyLoss(label_smoothing=0.3)(outputs, mixed_y)
            
            loss = loss + model.l1_reg()
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            batch_loss += loss.item()
            loss.backward()
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        scheduler.step()
        total_loss += batch_loss

    return total_loss / len(loader)

def evaluate(model, loader, config):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in loader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        
        crops = [
            F.interpolate(
                inputs.squeeze(-1).permute(0,3,1,2)[:,:,i:i+56,j:j+56], 
                size=(64,64), mode='bilinear', align_corners=False
            )
            for i,j in [(0,0), (8,8), (0,8), (8,0)]
        ]
        
        with torch.no_grad():
            outputs = torch.stack([
                model(crop.permute(0,2,3,1).unsqueeze(-1))
                for crop in crops
            ]).mean(0)
            
            loss = nn.CrossEntropyLoss()(outputs, labels)
            all_preds.append(outputs)
            all_labels.append(labels)
            total_loss += loss.item()

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    loss = total_loss / len(loader)
    
    return loss, {'accuracy': (preds.argmax(-1) == labels).float().mean().item()}

def main():
    logging.basicConfig(level=logging.INFO)
    seed_everything(42)

    # init
    config = Config()
    data_config = DataConfig()
    Path(config.CKPT_DIR).mkdir(parents=True, exist_ok=True)

    # data
    dataset_manager = DatasetManager(config, data_config)
    train_ds, val_ds, test_ds = dataset_manager.prepare_datasets()
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config)

    logger.info(f"samples/split: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # model
    model = VisionTransformerModel(config).to(config.device)
    verify_model_devices(model)

    # optim
    opt = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.1)
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    best_loss = float('inf')
    no_improve = 0

    # train loop w/ early stop
    for epoch in range(config.NUM_EPOCHS):
        if no_improve > 15:
            logger.info(f"plateau @ epoch {epoch}")
            break
            
        train_loss = train_one_epoch(model, train_dl, opt, sched, None, config)
        val_loss, val_metrics = evaluate(model, val_dl, config)

        logger.info(
            f"epoch {epoch:02d} - "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            save_checkpoint(model, opt, epoch, val_loss, config, "best_model.pth")
        else:
            no_improve += 1

    # final eval
    model = load_checkpoint(
        model, None, 
        Path(config.CKPT_DIR) / "best_model.pth",
        weights_only=True  
    )[0]
    
    test_loss, test_metrics = evaluate(model, test_dl, config)
    logger.info(f"test - loss:{test_loss:.4f} acc:{test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
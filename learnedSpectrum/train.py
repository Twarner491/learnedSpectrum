"""
Training script for fMRI Learning Stage Classification with Vision Transformers
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from learnedSpectrum.config import Config, DataConfig
from learnedSpectrum.data import DatasetManager, create_dataloaders
from learnedSpectrum.utils import (
    seed_everything,
    get_optimizer,
    verify_model_devices,
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    get_cosine_schedule_with_warmup
)
from transformers import ViTModel, ViTConfig

logger = logging.getLogger(__name__)

class VisionTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_drop = nn.Dropout2d(0.1)
        
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
            output_attentions=True  # crucial
        )
        self.vit = ViTModel(vit_config)
        
        self.classifiers = nn.ModuleList([
            nn.Linear(config.EMBED_DIM, config.NUM_CLASSES)
            for _ in range(3)
        ])
        
        self.l1_lambda = 1e-5  # l1 reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, return_attn=False):
        x = x.to(self.device).squeeze(-1).permute(0, 3, 1, 2)
        if x.shape[-1] != 64 or x.shape[-2] != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            
        x = self.spatial_drop(x)
        if self.training:
            x = x + torch.randn_like(x) * 0.01
            
        outputs = self.vit(pixel_values=x, output_attentions=return_attn)
        logits = torch.stack([
            clf(outputs.last_hidden_state[:, 0])
            for clf in self.classifiers
        ])
        
        if return_attn:
            return logits.mean(0), outputs.attentions[-1]
        return logits.mean(0)
        
    def l1_reg(self):
        """sparsity inducing reg"""
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return self.l1_lambda * l1_norm

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        batch_loss = 0
        optimizer.zero_grad()
        
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        
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
            loss = torch.nn.CrossEntropyLoss(
                label_smoothing=config.LABEL_SMOOTHING
            )(outputs, mixed_y)
            
            loss = loss + model.l1_reg()  # add l1
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            batch_loss += loss.item()
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
            
        total_loss += batch_loss

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        
        # simpler crops w/ interpolation
        crops = [
            F.interpolate(
                inputs.squeeze(-1).permute(0,3,1,2)[:,:,i:i+56,j:j+56], 
                size=(64,64), mode='bilinear', align_corners=False
            )
            for i,j in [(0,0), (8,8), (0,8), (8,0)]
        ]
        
        # stack predictions
        outputs = torch.stack([
            model(crop.permute(0,2,3,1).unsqueeze(-1))
            for crop in crops
        ]).mean(0)
        
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        
        total_loss += loss.item()
        all_outputs.append(outputs)
        all_labels.append(labels)
    
    outputs = torch.cat(all_outputs)
    labels = torch.cat(all_labels)
    
    metrics = calculate_metrics(outputs, labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics['loss'], metrics

def main():
    logging.basicConfig(level=logging.INFO)
    seed_everything(42)

    config = Config()
    data_config = DataConfig()
    
    # crucial: ensure checkpoint dir exists
    Path(config.CKPT_DIR).mkdir(parents=True, exist_ok=True)
    
    # load data w/ strict validation
    dataset_manager = DatasetManager(config, data_config)
    train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    logger.info(f"dataset sizes - train:{len(train_dataset)} val:{len(val_dataset)} test:{len(test_dataset)}")

    model = VisionTransformerModel(config).to(config.device)
    verify_model_devices(model)
    
    # crucial: track metrics
    best_metrics = {
        'val_loss': float('inf'),
        'val_acc': 0,
        'epoch': 0,
        'steps_no_improve': 0
    }
    
    optimizer = get_optimizer(model, config)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_EPOCHS * len(train_loader),
        num_training_steps=config.NUM_EPOCHS * len(train_loader)
    )
    scaler = GradScaler(enabled=config.USE_AMP)

    # training loop w/ early stopping
    for epoch in range(config.NUM_EPOCHS):
        if best_metrics['steps_no_improve'] > config.PATIENCE:
            logger.info(f"early stopping @ epoch {epoch}")
            break
            
        logger.info(f"epoch {epoch + 1}/{config.NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, config)
        
        if epoch % config.VAL_EVERY_N_EPOCHS == 0:
            val_loss, val_metrics = evaluate(model, val_loader, config)
            
            logger.info(
                f"train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} "
                f"val_acc:{val_metrics['accuracy']:.4f} val_auc:{val_metrics['auc']:.4f}"
            )
            
            # checkpoint handling
            if val_loss < best_metrics['val_loss']:
                best_metrics.update({
                    'val_loss': val_loss,
                    'val_acc': val_metrics['accuracy'],
                    'epoch': epoch,
                    'steps_no_improve': 0
                })
                save_checkpoint(
                    model, optimizer, epoch, val_loss, config, 
                    "best_model.pth"
                )
            else:
                best_metrics['steps_no_improve'] += 1
                
            # learning rate adjustment
            if best_metrics['steps_no_improve'] > config.WARMUP_EPOCHS:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
    
    # load best and eval
    logger.info(f"loading best model from epoch {best_metrics['epoch']}")
    model, _, _ = load_checkpoint(
        model, None, 
        Path(config.CKPT_DIR) / "best_model.pth"
    )
    
    test_loss, test_metrics = evaluate(model, test_loader, config)
    logger.info(
        f"final test - loss:{test_loss:.4f} acc:{test_metrics['accuracy']:.4f} "
        f"auc:{test_metrics['auc']:.4f}"
    )

if __name__ == "__main__":
    main()
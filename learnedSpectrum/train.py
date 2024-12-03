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
        self.dropout = nn.Dropout(0.5)  # aggressive dropout
        vit_config = ViTConfig(
            image_size=(config.VOLUME_SIZE[0], config.VOLUME_SIZE[1]),
            patch_size=config.PATCH_SIZE,
            num_channels=config.VOLUME_SIZE[2],
            hidden_size=config.EMBED_DIM // 2,  # halve capacity
            num_hidden_layers=config.DEPTH // 2,  # halve depth
            num_attention_heads=max(2, config.NUM_HEADS // 2),  # reduce heads
            intermediate_size=2 * config.EMBED_DIM,  # reduce mlp
            hidden_dropout_prob=0.3,  # increase dropout
            attention_probs_dropout_prob=0.3,
            num_labels=config.NUM_CLASSES
        )
        self.vit = ViTModel(vit_config)
        self.classifier = nn.Linear(config.EMBED_DIM // 2, config.NUM_CLASSES)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device).squeeze(-1).permute(0, 3, 1, 2)
        x = self.dropout(x)  # input dropout
        outputs = self.vit(pixel_values=x)
        return self.classifier(self.dropout(outputs.last_hidden_state[:, 0]))
    
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', 
                              enabled=config.USE_AMP):
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        if config.GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()  # critical for inference
def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    outputs, labels = [], []
    
    for batch in dataloader:
        x, y = [b.to(config.device) for b in batch]
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            out = model(x)
            loss = torch.nn.CrossEntropyLoss()(out, y)
        total_loss += loss.item()
        outputs.append(out)
        labels.append(y)
    
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
    
    return total_loss / len(dataloader), calculate_metrics(outputs, labels)

def main():
    logging.basicConfig(level=logging.INFO)
    seed_everything()

    config = Config()
    data_config = DataConfig()

    dataset_manager = DatasetManager(config, data_config)
    train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, config)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)

    model = VisionTransformerModel(config).to(config.device)
    verify_model_devices(model)

    optimizer = get_optimizer(model, config)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_EPOCHS * len(train_loader),
        num_training_steps=config.NUM_EPOCHS * len(train_loader)
    )
    scaler = GradScaler(enabled=config.USE_AMP)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, config)
        val_loss, val_metrics = evaluate(model, val_loader, config)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, "best_model.pth")
    logger.info("Training complete. Evaluating on test set...")
    test_loss, test_metrics = evaluate(model, test_loader, config)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}, Test AUC: {test_metrics['auc']:.4f}")
if __name__ == "__main__":
    main()
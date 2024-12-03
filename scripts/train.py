"""
Training script for fMRI Learning Stage Classification with Vision Transformers
"""

import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
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
    def __init__(self, config: Config):
        super(VisionTransformerModel, self).__init__()
        vit_config = ViTConfig(
            image_size=config.VOLUME_SIZE[0],
            patch_size=config.PATCH_SIZE,
            num_channels=config.VOLUME_SIZE[2],
            hidden_size=config.EMBED_DIM,
            num_hidden_layers=config.DEPTH,
            num_attention_heads=config.NUM_HEADS,
            intermediate_size=int(config.EMBED_DIM * config.MLP_RATIO),
            hidden_dropout_prob=config.DROP_RATE,
            attention_probs_dropout_prob=config.ATTN_DROP_RATE,
            num_labels=config.NUM_CLASSES
        )
        self.vit = ViTModel(vit_config)
        self.classifier = nn.Linear(config.EMBED_DIM, config.NUM_CLASSES)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = batch
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        with autocast(enabled=config.USE_AMP):
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        scaler.scale(loss).backward()

        if config.GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            with autocast(enabled=config.USE_AMP):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)

    return total_loss / len(dataloader), metrics

def main():
    logging.basicConfig(level=logging.INFO)
    seed_everything()

    config = Config()
    data_config = DataConfig()

    dataset_manager = DatasetManager(config, data_config)
    train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, config)

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
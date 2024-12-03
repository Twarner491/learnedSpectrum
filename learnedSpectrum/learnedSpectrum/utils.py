"""
Utility functions for LearnedSpectrum
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_gpu_memory() -> None:
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            free = total - reserved
            logger.info(f"GPU {i}: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved, {free:.1f}MB free of {total:.1f}MB total")

def get_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer with weight decay handling"""
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
            
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': config.WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=config.LEARNING_RATE)

def verify_model_devices(model: nn.Module) -> None:
    """Verify all model parameters are on the same device"""
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    if len(devices) > 1:
        raise RuntimeError(f"Model parameters are on different devices: {devices}")
    logger.info(f"Model is on device: {next(iter(devices))}")

def pretrain_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply pretraining transformations to input tensor"""
    # Normalize if not already done
    if x.mean() > 1e-3 or x.std() > 1:
        x = (x - x.mean()) / (x.std() + 1e-6)
    return x

def mixup(x: torch.Tensor, 
         y: torch.Tensor, 
         alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Perform mixup on the input batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   config,
                   filename: str) -> None:
    """Save model checkpoint"""
    checkpoint_path = Path(config.CKPT_DIR) / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   checkpoint_path: Union[str, Path]) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Dict]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return model, optimizer, checkpoint

def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                  num_warmup_steps: int,
                                  num_training_steps: int,
                                  num_cycles: float = 0.5,
                                  last_epoch: int = -1) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer."""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def calculate_metrics(outputs: torch.Tensor, 
                     targets: torch.Tensor) -> Dict[str, float]:
    """Calculate various metrics for model evaluation"""
    preds = outputs.argmax(dim=1)
    accuracy = (preds == targets).float().mean().item()
    
    # Convert to one-hot for ROC AUC
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.size(1))
    try:
        auc = roc_auc_score(
            targets_one_hot.cpu().numpy(),
            outputs.softmax(dim=1).cpu().numpy(),
            multi_class='ovr'
        )
    except ValueError:
        auc = 0.0
        
    return {
        'accuracy': accuracy,
        'auc': auc
    }

from typing import Dict, Tuple, Optional, Union
import logging
import random
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)


# foundation layer - no project deps
def seed_everything(seed: int = 42) -> None:
    """canonical torch seeding. fighting aleatoric uncertainty w/ epistemic certainty."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_memory() -> None:
    """debug memory pressure"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            free = total - reserved
            logger.info(f"GPU {i}: {allocated:.1f}MB alloc, {reserved:.1f}MB rsv, {free:.1f}MB free / {total:.1f}MB")


def get_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """adamw w/ smart decay grouping"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if len(param.shape) == 1 or name.endswith(".bias") else decay).append(param)
            
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': config.WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=config.LEARNING_RATE)


def verify_model_devices(model: nn.Module) -> None:
    """sanity check device placement"""
    devices = {param.device for param in model.parameters()}
    if len(devices) > 1:
        raise RuntimeError(f"model params scattered across: {devices}")
    logger.info(f"model on: {next(iter(devices))}")


def pretrain_transform(x: torch.Tensor) -> torch.Tensor:
    """basic normalization"""
    if x.mean() > 1e-3 or x.std() > 1:
        x = (x - x.mean()) / (x.std() + 1e-6)
    return x


def mixup(x: torch.Tensor, 
         y: torch.Tensor, 
         alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """mixup augmentation. zhang et al 2018."""
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
    """serialize model state"""
    checkpoint_path = Path(config.CKPT_DIR) / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    logger.info(f"checkpoint: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, weights_only=True):
    try:
        ckpt = torch.load(checkpoint_path, weights_only=weights_only)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return model, optimizer, ckpt
    except Exception as e:
        logging.warning(f"ckpt fail: {e}, continuing w/ fresh model")
        return model, optimizer, {}
    
    
def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                  num_warmup_steps: int,
                                  num_training_steps: int,
                                  num_cycles: float = 0.5) -> LambdaLR:
    """cosine decay w/ linear warmup. because cyclical > step."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)


def calculate_metrics(outputs: torch.Tensor, 
                     targets: torch.Tensor) -> Dict[str, float]:
    """basic clf metrics, nothing fancy"""
    preds = outputs.argmax(dim=1)
    acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
    
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.size(1))
    try:
        auc = roc_auc_score(
            targets_one_hot.cpu().numpy(),
            outputs.softmax(dim=1).cpu().numpy(),
            multi_class='ovr'
        )
    except ValueError:  # missing classes
        auc = float('nan')
    
    return {
        'accuracy': acc,
        'auc': auc,
    }
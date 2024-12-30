import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
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
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import os
import math
from torch.optim.lr_scheduler import LambdaLR

from learnedSpectrum.utils import (
    seed_everything, get_optimizer, verify_model_devices,
    calculate_metrics, save_checkpoint, load_checkpoint, print_gpu_memory,
    get_cosine_schedule_with_warmup, enable_memory_efficient_attention, checkpoint_wrapper
)
from learnedSpectrum.physiological import PhysiologicalStateTracker
from learnedSpectrum.causal import CausalAnalysisModule
from learnedSpectrum.visualization import TemporalUnderstandingVisualizer


logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0
    torch.cuda.empty_cache()  # Clear cache before training
    
    # Progress bar with better description
    pbar = tqdm(train_loader, desc=f'Training', 
                leave=False, dynamic_ncols=True)
    
    for batch_idx, (x, y) in enumerate(pbar):
        try:
            B = x.size(0)
            x = x.to(config.device, dtype=torch.float16, non_blocking=True)
            y = y.to(config.device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                # Forward pass with temporal state tracking
                logits, temporal_info = model(x)
                
                # Get temporal states with consistent dimensions
                current_states = temporal_info['temporal_states']
                
                # Compute losses
                ce_loss = F.cross_entropy(logits, y)
                temporal_loss = compute_temporal_consistency(current_states)
                hjb_loss = compute_hjb_loss(model, current_states, ce_loss, config)
                
                # Combined loss
                loss = (ce_loss + 
                       config.TEMPORAL_WEIGHT * temporal_loss +
                       config.HJB_WEIGHT * hjb_loss)
            
            # Optimization step
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.GRADIENT_CLIP_VAL
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / len(train_loader)


def evaluate(model, loader, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            try:
                inputs = inputs.to(config.device, dtype=config.AMP_DTYPE)
                labels = labels.to(config.device, dtype=torch.long)
                
                with torch.amp.autocast('cuda'):
                    outputs, _ = model(inputs)  # Unpack tuple
                    loss = F.cross_entropy(outputs, labels)
                    
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    # Calculate metrics
    metrics = calculate_validation_metrics(
        all_preds, all_labels, total_loss, len(loader), config
    )
    
    return metrics['loss'], metrics


def calculate_validation_metrics(all_preds, all_labels, total_loss, num_batches, config):
    if all_preds and all_labels:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Handle class imbalance and zero division
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': (all_preds == all_labels).mean(),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'precision': precision_score(
                all_labels, 
                all_preds, 
                average='weighted',
                zero_division=0
            ),
            'recall': recall_score(
                all_labels, 
                all_preds, 
                average='weighted',
                zero_division=0
            ),
            'f1': f1_score(
                all_labels, 
                all_preds, 
                average='weighted',
                zero_division=0
            ),
        }
        
        # Add prediction distribution analysis
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        metrics['pred_distribution'] = {
            f'class_{c}': count/len(all_preds) 
            for c, count in zip(unique_preds, pred_counts)
        }
        
        # Add label distribution
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        metrics['label_distribution'] = {
            f'class_{c}': count/len(all_labels)
            for c, count in zip(unique_labels, label_counts)
        }
        
    else:
        metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pred_distribution': {},
            'label_distribution': {}
        }
    
    return metrics


def train_loop(model, train_dl, val_dl, optimizer, config):
    # Enable memory efficient settings
    enable_memory_efficient_attention()
    
    # Replace model's forward with checkpointed version
    original_forward = model.forward
    model.forward = lambda *args, **kwargs: checkpoint_wrapper(
        original_forward, 
        *args, 
        **kwargs
    )
    
    # Verify model is on correct device
    verify_model_devices(model)
    
    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Calculate warmup steps
    num_training_steps = config.NUM_EPOCHS * len(train_dl)
    num_warmup_steps = int(0.1 * num_training_steps)

    # Initialize scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )

    # Initialize scaler
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        # Update current epoch
        config.current_epoch = epoch
        
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Training
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, config
        )

        # Validation
        val_loss, val_metrics = evaluate(model, val_dl, config)
        
        # Print detailed metrics
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {train_metrics:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
        print("\nPrediction Distribution:")
        for k, v in val_metrics['pred_distribution'].items():
            print(f"{k}: {v:.3f}")
        print("\nLabel Distribution:")
        for k, v in val_metrics['label_distribution'].items():
            print(f"{k}: {v:.3f}")
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, "best_model.pth")
            
        # Log metrics
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Train Loss: {train_metrics:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
    
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


def compute_temporal_understanding_loss(model_outputs, targets, config):
    # Unpack outputs
    predictions, temporal_info = model_outputs
    
    # Classification loss
    ce_loss = F.cross_entropy(predictions, targets)
    
    # Temporal consistency loss
    temporal_states = temporal_info['temporal_states']
    temporal_consistency = torch.tensor(0., device=predictions.device)
    for t1, t2 in zip(temporal_states[:-1], temporal_states[1:]):
        temporal_consistency += F.mse_loss(t1, t2)
    
    # HJB loss
    hjb_loss = model.compute_hjb_loss(
        temporal_states[-1],
        predictions,
        temporal_states[0],
        ce_loss.detach(),
        config.TEMPORAL_DT
    )
    
    # Causal loss
    causal_effects = temporal_info['causal_effects']
    causal_loss = compute_causal_consistency(causal_effects)
    
    # Combine losses
    total_loss = (
        ce_loss +
        config.TEMPORAL_WEIGHT * temporal_consistency +
        config.HJB_WEIGHT * hjb_loss +
        config.CAUSAL_WEIGHT * causal_loss
    )
    
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'temporal_loss': temporal_consistency.item(),
        'hjb_loss': hjb_loss.item(),
        'causal_loss': causal_loss.item()
    }


def compute_temporal_consistency(temporal_states):
    """
    Compute temporal consistency loss between sequential states
    """
    if not temporal_states:
        return torch.tensor(0.0, device=temporal_states[0].device)
        
    temporal_consistency = torch.tensor(0., device=temporal_states[0].device)
    
    # Compute MSE between consecutive states
    for t1, t2 in zip(temporal_states[:-1], temporal_states[1:]):
        # Handle potential dimension mismatches
        if t1.dim() != t2.dim():
            if t1.dim() == 2:
                t1 = t1.unsqueeze(1)
            if t2.dim() == 2:
                t2 = t2.unsqueeze(1)
                
        # Ensure same batch dimension
        if t1.size(0) != t2.size(0):
            min_batch = min(t1.size(0), t2.size(0))
            t1 = t1[:min_batch]
            t2 = t2[:min_batch]
            
        # Compute consistency loss
        temporal_consistency += F.mse_loss(t1, t2)
    
    # Average over number of transitions
    temporal_consistency = temporal_consistency / (len(temporal_states) - 1)
    
    return temporal_consistency


def compute_hjb_loss(model, temporal_states, ce_loss, config):
    """
    Compute Hamilton-Jacobi-Bellman loss for temporal understanding
    """
    if not temporal_states or len(temporal_states) < 2:
        return torch.tensor(0.0, device=ce_loss.device)
        
    # Get current and next states
    current_state = temporal_states[-2]  # Second to last state
    next_state = temporal_states[-1]     # Last state
    
    # Get temporal difference
    dt = torch.tensor(config.TEMPORAL_DT, device=current_state.device)
    
    # Compute HJB components using model's HJB solver
    if hasattr(model, 'hjb_solver'):
        action, value = model.hjb_solver(current_state, dt)
        next_value = model.hjb_solver(next_state, dt)[1]
        
        # Compute temporal difference
        value_diff = (next_value - value) / dt
        
        # Use CE loss as reward signal
        reward = -ce_loss.detach()
        
        # Compute HJB residual
        hjb_residual = (
            value_diff + 
            torch.mean(action**2, dim=-1, keepdim=True) / 2 + 
            reward - 
            config.HJB_GAMMA * value
        )
        
        return torch.mean(hjb_residual**2)
    else:
        # Fallback to simpler temporal consistency if HJB solver not available
        return F.mse_loss(current_state, next_state)
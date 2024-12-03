"""
Visualization utilities for fMRI data and training results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from nilearn import plotting
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Dict, Optional
import wandb

class VisualizationManager:
    def __init__(self, save_dir: Optional[Path] = None):
        if save_dir is None:
            save_dir = Path(__file__).parent.parent / "visualizations"
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_brain_slice(self, 
                        volume: np.ndarray,
                        slice_idx: int = None,
                        title: str = "Brain Slice",
                        save_name: Optional[str] = None):
        """Plot a single slice from a brain volume"""
        if slice_idx is None:
            slice_idx = volume.shape[-1] // 2

        plt.figure(figsize=(10, 8))
        plt.imshow(volume[:, :, slice_idx], cmap='gray')
        plt.colorbar()
        plt.title(title)
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()

    def plot_attention_map(self,
                          attention_weights: torch.Tensor,
                          volume_shape: tuple,
                          save_name: Optional[str] = None):
        """Visualize attention weights from the Vision Transformer"""
        # Reshape attention weights to match volume dimensions
        att_map = attention_weights.reshape(volume_shape)
        
        # Plot three orthogonal views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sagittal view
        axes[0].imshow(att_map[volume_shape[0]//2, :, :], cmap='hot')
        axes[0].set_title('Sagittal View')
        
        # Coronal view
        axes[1].imshow(att_map[:, volume_shape[1]//2, :], cmap='hot')
        axes[1].set_title('Coronal View')
        
        # Axial view
        axes[2].imshow(att_map[:, :, volume_shape[2]//2], cmap='hot')
        axes[2].set_title('Axial View')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
            
    def plot_training_history(self, history, save_name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_loss'], label='train')
        ax1.plot(history['val_loss'], label='val')
        ax1.set_title('loss')
        ax1.legend()
        
        ax2.plot(history['train_acc'], label='train')
        ax2.plot(history['val_acc'], label='val')
        ax2.set_title('accuracy')
        ax2.legend()
        
        plt.savefig(self.save_dir / f"{save_name}.png")
        plt.close()

    def log_to_wandb(self, metrics, step):
        """wandb metric dump"""
        wandb.log(metrics, step=step)
        
        if 'train_metrics' in metrics:
            # flatten nested dicts
            for k, v in metrics['train_metrics'].items():
                wandb.log({f'train_{k}': v}, step=step)
        if 'val_metrics' in metrics:
            for k, v in metrics['val_metrics'].items():
                wandb.log({f'val_{k}': v}, step=step)

    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            classes: List[str],
                            save_name: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()
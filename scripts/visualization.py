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

    def plot_training_history(self,
                            history: Dict[str, List[float]],
                            save_name: Optional[str] = None):
        """Plot training metrics history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()

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

    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_scores: np.ndarray,
                       classes: List[str],
                       save_name: Optional[str] = None):
        """Plot ROC curves for each class"""
        plt.figure(figsize=(10, 8))
        
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show()

    def log_to_wandb(self,
                    metrics: Dict[str, float],
                    step: int,
                    prefix: str = ""):
        """Log metrics to Weights & Biases"""
        if wandb.run is not None:
            wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def plot_brain_activation(self,
                            volume: np.ndarray,
                            background_img: Optional[str] = None,
                            save_name: Optional[str] = None):
        """Plot brain activation using nilearn"""
        if background_img is None:
            background_img = 'MNI152'
            
        display = plotting.plot_stat_map(
            volume,
            bg_img=background_img,
            display_mode='ortho',
            cut_coords=(0, 0, 0),
            colorbar=True,
            title='Brain Activation'
        )
        
        if save_name and self.save_dir:
            display.savefig(self.save_dir / f"{save_name}.png")
            plt.close()
        else:
            plt.show() 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import wandb
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class VisualizationManager:
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path().absolute() / "notebooks/visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # modern style config
        plt.style.use('dark_background')  # base
        plt.rcParams.update({
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': '#1f1f1f',
            'figure.facecolor': '#2d2d2d',
            'axes.titlesize': 10,
            'axes.labelsize': 9
        })
        
    def save_or_show(self, save_name: Optional[str] = None):
        """handle io w/ proper cleanup"""
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
    def plot_brain_slice(self, volume: np.ndarray,
                        slice_idx: Optional[int] = None,
                        time_idx: Optional[int] = None,
                        title: str = "brain slice",
                        save_name: Optional[str] = None,
                        cmap: str = 'gray'):
        """4d fmri vis w/ proper bounds"""
        if len(volume.shape) == 4:
            time_idx = time_idx or volume.shape[-1] // 2
            volume = volume[..., time_idx]
            
        slice_idx = slice_idx or volume.shape[2] // 2
        slice_idx = min(slice_idx, volume.shape[2] - 1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(volume[:, :, slice_idx], cmap=cmap)
        plt.colorbar(label='intensity')
        plt.title(f"{title} [z={slice_idx}, t={time_idx}]")
        self.save_or_show(save_name)

    def plot_attention_map(self, attention_weights: torch.Tensor,
                         volume_shape: Tuple[int, ...],
                         save_name: Optional[str] = None):
        """attn map vis w/ proper normalization"""
        try:
            if torch.is_tensor(attention_weights):
                att_map = attention_weights.detach().cpu().numpy()
            else:
                att_map = attention_weights
                
            # handle cls token properly
            att_map = att_map[:, 1:].mean(0)  # avg heads, rm cls
            att_map = att_map.reshape(volume_shape)
            
            views = ['sagittal', 'coronal', 'axial']
            mid_pts = [s//2 for s in volume_shape]
            slices = [att_map[mid_pts[0], :, :],
                     att_map[:, mid_pts[1], :],
                     att_map[:, :, mid_pts[2]]]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, view, slice_data in zip(axes, views, slices):
                im = ax.imshow(slice_data, cmap='inferno')
                ax.set_title(f'{view}')
                plt.colorbar(im, ax=ax)
                
            plt.suptitle('attention distribution')
            plt.tight_layout()
            self.save_or_show(save_name)
            
        except Exception as e:
            logger.error(f"attn viz fail: {str(e)}")
            plt.close()

    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_name: Optional[str] = None):
        """learning curves w/ proper scaling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        for metric, ax in [('loss', ax1), ('acc', ax2)]:
            for split in ['train', 'val']:
                key = f'{split}_{metric}'
                if key in history and len(history[key]) > 0:
                    ax.plot(history[key], label=split)
                    
            ax.set_yscale('log' if metric == 'loss' else 'linear')
            ax.set_title(metric)
            ax.set_xlabel('epoch')
            ax.legend()
            
        plt.tight_layout()
        self.save_or_show(save_name)

    def plot_confusion_matrix(self, y_true: np.ndarray,
                            y_pred: np.ndarray,
                            classes: List[str],
                            save_name: Optional[str] = None,
                            normalize: bool = True):
        """conf matrix w/ proper handling of class imbalance"""
        if len(np.unique(y_true)) == 1:
            logger.warning("degenerate predictions detected")
            return
            
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, 
                   fmt='.2%' if normalize else 'd',
                   cmap='Blues',
                   xticklabels=classes, 
                   yticklabels=classes)
        plt.title('confusion matrix' + 
                 (' (normalized)' if normalize else ''))
        plt.xlabel('predicted')
        plt.ylabel('true')
        self.save_or_show(save_name)

    def plot_roc_curves(self, y_true: np.ndarray,
                       y_scores: np.ndarray,
                       classes: List[str],
                       save_name: Optional[str] = None):
        """roc curves w/ proper multiclass handling"""
        plt.figure(figsize=(10, 8))
        
        if len(np.unique(y_true)) <= 1:
            logger.warning("insufficient classes for roc")
            return
            
        try:
            y_bin = F.one_hot(
                torch.tensor(y_true), 
                num_classes=len(classes)
            ).numpy()
            
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{cls} (auc={roc_auc:.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--', label='random')
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title('roc curves per class')
            plt.legend()
            self.save_or_show(save_name)
            
        except Exception as e:
            logger.error(f"roc fail: {str(e)}")
            plt.close()

    def plot_gradient_flow(self, model: torch.nn.Module,
                          save_name: Optional[str] = None):
        """gradient health monitor"""
        named_params = [
            (n,p) for n,p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        
        plt.figure(figsize=(12, 6))
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in named_params:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
            
        plt.semilogy(ave_grads, 'b', label='mean')
        plt.semilogy(max_grads, 'r', label='max')
        plt.xticks(range(len(layers)), layers, 
                  rotation=45, ha='right')
        plt.grid(True)
        plt.legend()
        plt.title('gradient magnitude distribution')
        plt.tight_layout()
        self.save_or_show(save_name)

    def log_to_wandb(self, metrics: Dict, step: int):
        """wandb logging w/ proper metric handling"""
        try:
            wandb.log(metrics, step=step)
            for split in ['train', 'val']:
                if f'{split}_metrics' in metrics:
                    for k, v in metrics[f'{split}_metrics'].items():
                        if not np.isnan(v):  # filter bad metrics
                            wandb.log({f'{split}_{k}': v}, step=step)
        except Exception as e:
            logger.error(f"wandb fail: {str(e)}")

    def plot_prediction_distribution(self, probs: np.ndarray,
                                   labels: np.ndarray,
                                   save_name: Optional[str] = None):
        """confidence distribution analysis"""
        plt.figure(figsize=(10, 6))
        
        for i in range(probs.shape[1]):
            mask = labels == i
            if mask.any():
                sns.kdeplot(probs[mask, i],
                           label=f'class {i} (n={mask.sum()})')
                
        plt.axvline(0.5, color='k', linestyle='--',
                   alpha=0.3, label='decision boundary')
        plt.title('prediction confidence distribution')
        plt.xlabel('model confidence')
        plt.ylabel('density')
        plt.legend()
        self.save_or_show(save_name)
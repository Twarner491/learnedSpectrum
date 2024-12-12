import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import wandb
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import logging
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class VisualizationManager:
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path().absolute() / "notebooks/visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Scientific publication style with rainbow color palette
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        plt.rcParams.update({
            'axes.prop_cycle': plt.cycler(color=self.colors),
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.2,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'font.family': 'Arial',
            'figure.dpi': 300,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black'
        })
        
    def save_or_show(self, save_name: Optional[str] = None):
        try:
            if save_name:
                save_path = self.save_dir / f"{save_name}.png"
                self.save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed to save/show plot: {str(e)}")
        finally:
            plt.close('all')  # Ensure all figures are closed

    def plot_brain_slice(self, volume: np.ndarray,
                        slice_idx: Optional[int] = None,
                        time_idx: Optional[int] = None,
                        title: str = "brain slice",
                        save_name: Optional[str] = None,
                        cmap: str = 'turbo'):
        """4d fmri vis w/ proper bounds"""
        if len(volume.shape) == 4:
            time_idx = time_idx or volume.shape[-1] // 2
            volume = volume[..., time_idx]
            
        slice_idx = slice_idx or volume.shape[2] // 2
        slice_idx = min(slice_idx, volume.shape[2] - 1)
        
        # Create higher resolution figure with white background
        fig = plt.figure(figsize=(10, 8), facecolor='white', dpi=300)  # Increased size and DPI
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Plot the slice with improved styling and interpolation
        im = ax.imshow(volume[:, :, slice_idx], 
                      cmap=cmap,
                      interpolation='bicubic',  # Changed to bicubic for smoother interpolation
                      aspect='equal')
        
        # Add a clean colorbar
        cbar = plt.colorbar(im, label='Intensity', ax=ax)
        cbar.ax.tick_params(labelsize=10)  # Slightly larger ticks for higher res
        cbar.ax.yaxis.label.set_size(11)
        
        # Clean up the axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add title with proper formatting
        plt.title(f"{title}\n[z={slice_idx}" + (f", t={time_idx}]" if time_idx is not None else "]"),
                  pad=10, fontsize=14)  # Slightly larger title for higher res
        
        plt.tight_layout()
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
                
            # Get the actual size of attention weights
            att_size = att_map.size if isinstance(att_map, np.ndarray) else att_map.numel()
            logger.info(f"Attention map size: {att_size}, target shape: {volume_shape}")
            
            # Safely reshape or pad/truncate to match volume shape
            total_voxels = np.prod(volume_shape)
            if att_size != total_voxels:
                logger.warning(f"Attention map size mismatch: {att_size} vs {total_voxels}")
                # Pad or truncate to match
                att_map = np.pad(att_map.flatten(), 
                               (0, max(0, total_voxels - att_size)))[:total_voxels]
                
            # Reshape to volume shape
            att_map = att_map.reshape(volume_shape)
            
            views = ['sagittal', 'coronal', 'axial']
            mid_pts = [s//2 for s in volume_shape]
            slices = [att_map[mid_pts[0], :, :],
                     att_map[:, mid_pts[1], :],
                     att_map[:, :, mid_pts[2]]]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, view, slice_data in zip(axes, views, slices):
                im = ax.imshow(slice_data, cmap='viridis')
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
        """Plot training metrics with scientific styling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        metrics = [('loss', ax1), ('acc', ax2)]
        splits = ['train', 'val']
        
        for metric, ax in metrics:
            for i, split in enumerate(splits):
                key = f'{split}_{metric}'
                if key in history and len(history[key]) > 0:
                    ax.plot(history[key], 
                           color=self.colors[i],
                           label=split.capitalize(),
                           linewidth=2)
            
            ax.set_yscale('log' if metric == 'loss' else 'linear')
            ax.set_title(f'{metric.capitalize()}', pad=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False)
        
        plt.tight_layout()
        self.save_or_show(save_name)

    def plot_confusion_matrix(self, y_true: np.ndarray,
                            y_pred: np.ndarray,
                            classes: List[str],
                            save_name: Optional[str] = None,
                            normalize: bool = True):
        try:
            plt.close('all')
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            if len(unique_classes) == 1:
                logger.warning(f"Only one class present: {classes[unique_classes[0]]}")
                fig = plt.figure(figsize=(6, 6))
                plt.text(0.5, 0.5, f"All predictions: {classes[unique_classes[0]]}", 
                        ha='center', va='center', color='black')
                plt.axis('off')
            else:
                cm = confusion_matrix(y_true, y_pred)
                if normalize:
                    cm = cm.astype('float32') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
                
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, 
                           fmt='.2%' if normalize else 'd',
                           cmap='YlOrRd',
                           xticklabels=classes, 
                           yticklabels=classes,
                           cbar_kws={'label': 'Proportion' if normalize else 'Count'})
                plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
                plt.xlabel('Predicted')
                plt.ylabel('True')
            
            self.save_or_show(save_name)
            
        except Exception as e:
            logger.error(f"Confusion matrix plotting failed: {str(e)}")
            plt.close('all')

    def plot_roc_curves(self, y_true: np.ndarray,
                       y_scores: np.ndarray,
                       classes: List[str],
                       save_name: Optional[str] = None):
        try:
            plt.close('all')
            if len(np.unique(y_true)) <= 1:
                logger.warning("Insufficient classes for ROC curve")
                return
                
            fig = plt.figure(figsize=(8, 6))  # Reduced figure size
            
            # Convert to float32 for memory efficiency
            y_scores = y_scores.astype(np.float32)
            
            y_bin = F.one_hot(
                torch.tensor(y_true), 
                num_classes=len(classes)
            ).numpy()
            
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves by Class')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            self.save_or_show(save_name)
            
        except Exception as e:
            logger.error(f"ROC curve plotting failed: {str(e)}")
            plt.close('all')

    def plot_gradient_flow(self, model: torch.nn.Module,
                          save_name: Optional[str] = None):
        """Gradient flow analysis with scientific styling"""
        named_params = [(n,p) for n,p in model.named_parameters() 
                       if p.requires_grad and p.grad is not None]
        
        plt.figure(figsize=(12, 6))
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in named_params:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
            
        plt.semilogy(ave_grads, color=self.colors[0], 
                     label='Mean', linewidth=2)
        plt.semilogy(max_grads, color=self.colors[1], 
                     label='Max', linewidth=2)
        
        plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
        plt.grid(True, alpha=0.2)
        plt.legend(frameon=False)
        plt.title('Gradient Flow Analysis')
        plt.xlabel('Layers')
        plt.ylabel('Gradient Magnitude (log scale)')
        
        plt.tight_layout()
        self.save_or_show(save_name)

    def log_to_wandb(self, metrics: Dict, step: int):
        try:
            # Ensure metrics are JSON serializable
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)) and not np.isnan(v):
                    clean_metrics[k] = v
                elif isinstance(v, dict):
                    clean_metrics[k] = {
                        sk: sv for sk, sv in v.items() 
                        if isinstance(sv, (int, float, str, bool)) and not np.isnan(sv)
                    }
            
            wandb.log(clean_metrics, step=step)
            
        except Exception as e:
            logger.error(f"WandB logging failed: {str(e)}")

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

    def update_visualization_style(self):
        """Update style for publication quality"""
        plt.style.use('default')
        plt.rcParams.update({
            'axes.grid': True,
            'grid.alpha': 0.2,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'figure.dpi': 300,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
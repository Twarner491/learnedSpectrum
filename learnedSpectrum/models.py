import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Tuple
import math
from .hjb import ContinuousTimeQLearning
from .causal import CausalEstimator, TemporalCausalConv
import numpy as np

from learnedSpectrum.hjb import HJBSolver


class CustomChannelReduce(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv3d(64, 32, 1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 16, 1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Dropout3d(0.1)
        )
    
    def forward(self, x):
        x = x.permute(0, 1, 2, 3, 4)
        x = self.reduce(x)
        return x
    

class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.GELU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out
    

def checkpoint_forward(module, *args, **kwargs):
    """Wrapper for consistent checkpoint behavior"""
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    
    # Enable gradient tracking for inputs that need it
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and not arg.requires_grad:
            args[i] = arg.detach().requires_grad_(True)
    
    return torch.utils.checkpoint.checkpoint(
        custom_forward,
        *args,
        use_reentrant=False,
        preserve_rng_state=False
    )


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=drop,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Add state projection to ensure consistent dimensions
        self.state_proj = nn.Linear(dim, dim)
        
    def forward(self, x, h=None, dt=None):
        if self.training:
            return checkpoint_forward(self._forward_impl, x, h, dt)
        return self._forward_impl(x, h, dt)
    
    def _forward_impl(self, x, h=None, dt=None):
        # Apply first normalization and attention
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, x_norm, x_norm)[0]
        x = x + attn_out
        
        # Apply second normalization and MLP
        x = x + self.mlp(self.norm2(x))
        
        # Project state to consistent dimension
        h_new = self.state_proj(x.mean(dim=1))  # [B, D]
        
        return x, h_new


class LTCLayer(nn.Module):
    def __init__(self, hidden_dim, tau_min=1.0, tau_max=100.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Enhanced LTC cell
        self.ltc_cell = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Adaptive tau network
        self.tau_net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Add state normalization
        self.state_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, h, dt):
        # Normalize states
        x = self.state_norm(x)
        h = self.state_norm(h) if h is not None else torch.zeros_like(x)
        
        # Ensure consistent dimensions
        if h.size(0) != x.size(0):
            # Properly handle 3D tensors
            if h.dim() == 3:
                h = h.expand(x.size(0), h.size(1), h.size(2))
            else:
                h = h.expand(x.size(0), h.size(1))
            
        # Ensure matching dimensions for concatenation
        if h.dim() != x.dim():
            if h.dim() == 2 and x.dim() == 3:
                h = h.unsqueeze(1).expand(-1, x.size(1), -1)
            elif h.dim() == 3 and x.dim() == 2:
                h = h.mean(dim=1)  # Or another reduction strategy
                
        combined = torch.cat([x, h], dim=-1)
        dh = self.ltc_cell(combined)
        
        # Compute adaptive time constants
        tau = self.tau_min + (self.tau_max - self.tau_min) * self.tau_net(h)
        
        # Update state with temporal consistency
        if dt.dim() < h.dim():
            dt = dt.view(-1, 1, 1) if h.dim() == 3 else dt.view(-1, 1)
            
        h_new = h + (dh - h) * dt / tau
        
        return h_new


class EnhancedTemporalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.transformer = TransformerBlock(dim, num_heads, mlp_ratio, drop)
        self.ltc = LTCLayer(dim)
        
    def forward(self, x, h=None, dt=None):
        # Default dt if not provided
        if dt is None:
            dt = torch.tensor(0.1, device=x.device)
            
        # Transformer processing
        x = self.transformer(x)
        
        # Initialize hidden state if None
        B, N, D = x.shape
        if h is None:
            h = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
            
        # Process through LTC
        h_new = self.ltc(x, h, dt)
        
        return x, h_new


class VisionTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Channel reduction and temporal processing
        self.config = config
        self.channel_reduce = CustomChannelReduce()
        self.temporal_net = AdaptiveTemporalProcessor(
            config.EMBED_DIM//2, 
            config.NUM_HEADS//2,
            config.TEMPORAL_DIM
        )
        
        # Calculate patch dimensions
        H_out = (config.VOLUME_SIZE[0] + config.PATCH_SIZE) // (config.PATCH_SIZE//2) - 1
        W_out = (config.VOLUME_SIZE[1] + config.PATCH_SIZE) // (config.PATCH_SIZE//2) - 1
        self.num_patches = H_out * W_out
        
        # Initialize position embedding with a temporary size
        # It will be updated in _init_pos_embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, config.EMBED_DIM//2)  # Start with minimal size
        )
        
        # Initialize patch embedding with corrected LayerNorm
        self.patch_embed = nn.Sequential(
            nn.Conv2d(16, config.EMBED_DIM//2, 
                      kernel_size=config.PATCH_SIZE, 
                      stride=config.PATCH_SIZE//2,
                      padding=config.PATCH_SIZE//2),
            nn.Flatten(2),
            Permute(0, 2, 1),  # Transpose before LayerNorm
            nn.LayerNorm(config.EMBED_DIM//2),  # Now operates on last dimension
            nn.GELU()
        )
        
        # Position embedding and CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.EMBED_DIM//2))
        
        # Now initialize position embeddings
        self._init_pos_embed()
        
        # Transformer blocks with correct dimensions
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.config.EMBED_DIM//2,  # Match the embedding dimension
                num_heads=self.config.NUM_HEADS,
                mlp_ratio=4.0,
                drop=0.1
            ) for _ in range(self.config.DEPTH)
        ])
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(config.EMBED_DIM//2),
            nn.Linear(config.EMBED_DIM//2, config.NUM_CLASSES)
        )
        
    def _init_pos_embed(self):
        # Calculate expected sequence length based on input dimensions
        H = (self.config.VOLUME_SIZE[0] // self.config.PATCH_SIZE) + 1
        W = (self.config.VOLUME_SIZE[1] // self.config.PATCH_SIZE) + 1
        self.num_patches = H * W
        
        # Initialize position embedding with correct size
        embed_dim = self.config.EMBED_DIM//2
        grid_size = int(math.sqrt(self.num_patches))
        
        # Generate position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=grid_size,
            cls_token=True
        )
        
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
    def forward(self, x):
        B, C_in, H_in, W_in, T = x.shape
        chunk_size = min(T, self.config.TEMPORAL_CHUNK_SIZE)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Channel reduction
            x = self.channel_reduce(x)
            
            # Initialize temporal state
            h_temporal = None
            dt = torch.tensor(self.config.TEMPORAL_DT, device=x.device)
            
            # Temporal processing
            x, h_temporal = self.temporal_net(x, h_temporal, dt)
            
            _, C, H, W, _ = x.shape
            temporal_states = []
            chunk_embeddings = []
            
            # Process temporal chunks with state tracking
            h = None
            for t in range(0, T, chunk_size):
                end_t = min(t + chunk_size, T)
                x_chunk = x[:, :, :, :, t:end_t].contiguous()
                
                # Process spatial information
                chunk = x_chunk.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
                chunk = self.patch_embed(chunk)
                
                # Add cls token
                cls_tokens = self.cls_token.expand(chunk.shape[0], -1, -1)
                chunk = torch.cat((cls_tokens, chunk), dim=1)
                
                # Interpolate position embeddings if needed
                if chunk.shape[1] != self.pos_embed.shape[1]:
                    pos_embed = interpolate_pos_embed(
                        pos_embed=self.pos_embed,
                        target_size=chunk.shape[1] - 1,  # Subtract 1 for cls token
                        num_prefix_tokens=1
                    )
                else:
                    pos_embed = self.pos_embed
                    
                chunk = chunk + pos_embed
                
                # Process through transformer blocks
                for block in self.blocks:
                    chunk, h = block(chunk, h, dt)
                    temporal_states.append(h)
                
                # Reshape embeddings to maintain consistent size
                cls_embedding = chunk[:, 0]  # Get CLS token embeddings
                cls_embedding = cls_embedding.view(B, -1, cls_embedding.size(-1))  # Reshape to [B, T', D]
                chunk_embeddings.append(cls_embedding)
            
            # Combine embeddings with proper padding
            max_length = max(emb.size(1) for emb in chunk_embeddings)
            padded_embeddings = []
            
            for emb in chunk_embeddings:
                if emb.size(1) < max_length:
                    padding = torch.zeros(B, max_length - emb.size(1), emb.size(-1), 
                                       device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, padding], dim=1)
                padded_embeddings.append(emb)
            
            embeddings = torch.cat(padded_embeddings, dim=1)
            logits = self.head(embeddings.mean(dim=1))
            
            return logits, {'temporal_states': temporal_states}


class AdaptiveTemporalProcessor(nn.Module):
    def __init__(self, dim, num_heads, temporal_dim):
        super().__init__()
        self.input_proj = nn.Linear(16, dim)
        self.temporal_attn = TransformerBlock(dim, num_heads)
        self.ltc = LTCLayer(dim)
        self.hjb = HJBSolver(dim)
        self.causal_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=8)  # Use grouped conv
        self.output_proj = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, 16)
        )
        
    def forward(self, x, h, dt):
        B, C, H, W, T = x.shape
        
        # Process in smaller chunks with gradient checkpointing
        chunk_size = 16  # Reduced chunk size
        x_flat = x.permute(0, 4, 1, 2, 3).reshape(B*T, C, H*W)
        x_flat = x_flat.transpose(1, 2)
        
        outputs = []
        for i in range(0, B*T, chunk_size):
            end_idx = min(i + chunk_size, B*T)
            chunk = x_flat[i:end_idx]
            
            # Free memory
            if i > 0:
                del temporal_repr, ltc_state, causal_features, temporal_features
                torch.cuda.empty_cache()
            
            # Use gradient checkpointing for memory efficiency
            chunk = self.input_proj(chunk)
            temporal_repr, _ = torch.utils.checkpoint.checkpoint(self.temporal_attn, chunk)
            
            if h is None:
                h_chunk = torch.zeros(
                    (1, temporal_repr.size(1), temporal_repr.size(2)), 
                    device=temporal_repr.device, 
                    dtype=temporal_repr.dtype
                )
            else:
                h_chunk = h[i:end_idx]
            
            ltc_state = self.ltc(temporal_repr, h_chunk, dt)
            
            # Process features with memory optimization
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                causal_features = self.causal_estimator(temporal_repr)
                conv_input = temporal_repr.transpose(1, 2)
                temporal_features = self.temporal_conv(conv_input).transpose(1, 2)
            
            # Combine and project
            combined = torch.cat([
                temporal_repr,
                ltc_state,
                causal_features,
                temporal_features
            ], dim=-1)
            
            output = self.output_proj(combined)
            outputs.append(output.detach())  # Detach to save memory
            
        # Combine results
        output = torch.cat(outputs, dim=0)
        output = output.transpose(1, 2)
        output = output.reshape(B, T, 16, H, W)
        output = output.permute(0, 2, 3, 4, 1)
        
        return output, outputs[-1]


class TemporalCausalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        # Use same padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=16,  # Match input dimension
            out_channels=16,  # Match output dimension
            kernel_size=kernel_size,
            padding=padding  # Use same padding
        )
        self.norm = nn.BatchNorm1d(16)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x shape: [B*T, N, D]
        # Transpose for conv1d: [B*T, D, N]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        # Transpose back: [B*T, N, D]
        return x.transpose(1, 2)
    

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    embed_dim: output dimension for each position
    grid: numpy array of shape [2, 1, H, W]
    """
    assert embed_dim % 2 == 0
    
    # Flatten grid to [H*W, 2]
    grid = grid.reshape([2, -1]).transpose([1, 0])
    
    # Get embeddings for each dimension
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 1])
    
    # Combine embeddings
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: numpy array of positions to be encoded
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def interpolate_pos_embed(pos_embed, target_size, num_prefix_tokens=1):
    """Interpolate position embeddings to target size."""
    pos_tokens = pos_embed[:, num_prefix_tokens:] # exclude cls token
    pos_tokens = pos_tokens.transpose(1, 2)
    pos_tokens = F.interpolate(
        pos_tokens.unsqueeze(0), 
        size=(target_size),
        mode='linear',
        align_corners=False
    )
    pos_tokens = pos_tokens.squeeze(0).transpose(1, 2)
    
    # Restore cls token
    pos_embed = torch.cat([pos_embed[:, :num_prefix_tokens], pos_tokens], dim=1)
    return pos_embed


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x) 


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims) 
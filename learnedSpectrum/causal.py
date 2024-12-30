from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from dataclasses import dataclass
from torch.distributions import Normal

@dataclass
class CausalVariable:
    name: str
    type: str  
    temporal_lag: int = 0
    is_observed: bool = True


class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.variables: Dict[str, CausalVariable] = {}
        
    def add_variable(self, var: CausalVariable):
        self.variables[var.name] = var
        self.graph.add_node(var.name, 
                          type=var.type,
                          temporal_lag=var.temporal_lag,
                          is_observed=var.is_observed)
        
    def add_edge(self, cause: str, effect: str):
        if cause not in self.variables or effect not in self.variables:
            raise ValueError("Variables must be added before creating edges")
        self.graph.add_edge(cause, effect)
        
    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        return [p for p in nx.all_simple_paths(self.graph, treatment, outcome)
                if self._is_backdoor_path(p)]
                
    def _is_backdoor_path(self, path: List[str]) -> bool:
        return len(path) > 2 and all(
            self.variables[node].type == 'confounder' 
            for node in path[1:-1]
        )
    

class CausalEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Dimension reduction for memory efficiency
        self.dim_reduce = nn.Sequential(
            nn.Linear(16, hidden_dim),  # Match input dimension
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Treatment effect estimation
        self.treatment_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 16)  # Match output dimension
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Debug prints
        print(f"Input shape: {x.shape}")
        
        # Reshape if needed
        if len(x.shape) > 3:
            B, T, H, W = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            print(f"Reshaped input: {x.shape}")
        
        x = self.dim_reduce(x)
        print(f"After dim reduce: {x.shape}")
        
        features = self.feature_net(x)
        print(f"After feature net: {features.shape}")
        
        treatment_effects = self.treatment_net(features)
        print(f"Treatment effects: {treatment_effects.shape}")
        
        # Return treatment effects and features
        return treatment_effects, features


class TemporalCausalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                            padding=(kernel_size - 1))
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.register_buffer('mask', self._get_causal_mask(kernel_size))
        
    def _get_causal_mask(self, kernel_size: int) -> torch.Tensor:
        mask = torch.ones(kernel_size)
        mask[kernel_size//2 + 1:] = 0
        return mask.view(1, 1, -1)  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, T = x.shape
        
        x = x.permute(0, 1, 4, 2, 3)  
        x = x.reshape(B * C, T, H * W)  
        x = x.permute(0, 2, 1)  
        
        x = self.conv(x) * self.mask
        x = self.norm(x)
        x = self.activation(x)
        
        x = x.permute(0, 2, 1)  
        x = x.reshape(B, C, T, H, W)  
        x = x.permute(0, 1, 3, 4, 2)  
        
        return x


class CausalAnalysisModule(nn.Module):
    def __init__(self, feature_dim: int, temporal_dim: int):
        super().__init__()
        
        self.temporal_encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.causal_estimator = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, 1),
            nn.Sigmoid()
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=temporal_dim * 2,
            num_heads=4,
            batch_first=True
        )
        
    def compute_causal_strength(self, x: torch.Tensor, 
                              temporal_mask: Optional[torch.Tensor] = None):
        # Encode temporal sequence
        temporal_features, _ = self.temporal_encoder(x)
        
        # Apply temporal attention for causal discovery
        if temporal_mask is not None:
            attended_features, _ = self.temporal_attention(
                temporal_features, temporal_features, temporal_features,
                key_padding_mask=temporal_mask
            )
        else:
            attended_features, _ = self.temporal_attention(
                temporal_features, temporal_features, temporal_features
            )
            
        # Estimate causal strengths
        causal_strengths = self.causal_estimator(attended_features)
        
        return causal_strengths, attended_features
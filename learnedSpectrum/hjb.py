import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class HJBValueNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hamiltonian = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        state_features = self.state_encoder(state)
        value = self.value_net(state_features)
        
        if action is not None:
            ham_input = torch.cat([state_features, action], dim=-1)
            hamiltonian = self.hamiltonian(ham_input)
            return value, hamiltonian
        
        return value, None
    

class ContinuousTimeQLearning(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        super().__init__()
        
        self.gamma = gamma
        self.value_net = HJBValueNetwork(state_dim, action_dim)

        self.td_net = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def compute_hjb_loss(self, state: torch.Tensor, action: torch.Tensor, 
                        next_state: torch.Tensor, reward: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        value, hamiltonian = self.value_net(state, action)

        next_value, _ = self.value_net(next_state)

        temporal_diff = (next_value - value) / dt
        value_gradient = torch.autograd.grad(value.sum(), state, create_graph=True)[0]

        hjb_residual = temporal_diff + hamiltonian + reward - self.gamma * value
        
        return torch.mean(hjb_residual**2)
    
    def forward(self, state: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.policy_net(state)

        value, hamiltonian = self.value_net(state, action)
        
        return action, value 

class HJBSolver(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # Enhanced value network
        self.value_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim * 2, 1)
        )
        
        # Enhanced action network
        self.action_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim * 2, state_dim)
        )
        
        # Add temporal consistency layer
        self.temporal_consistency = nn.GRU(
            state_dim, state_dim, 
            num_layers=1, 
            batch_first=True
        )
        
    def forward(self, x, dt):
        batch_size = x.size(0)
        
        # Ensure consistent dimensions
        if x.dim() == 3:  # [B, T, D]
            x_flat = x.reshape(-1, self.state_dim)
        else:
            x_flat = x
            
        # Compute value and action
        value = self.value_net(x_flat)
        action = self.action_net(x_flat)
        
        # Apply temporal consistency
        if x.dim() == 3:
            _, h = self.temporal_consistency(x)
            action = action.view(batch_size, -1, self.state_dim)
            value = value.view(batch_size, -1, 1)
            
        # Scale action by dt
        action = action * dt.view(-1, 1)
        
        return action, value 
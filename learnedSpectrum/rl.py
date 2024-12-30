import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class TemporalStateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(x)


class TemporalQLearning(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = TemporalStateEncoder(state_dim, hidden_dim)
        
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        self.update_target_network(tau=1.0)
        
    def update_target_network(self, tau: float = 0.005):
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )
            
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_features = self.state_encoder(state)
        return self.q_network(state_features)
    
    def get_td_error(self, state: torch.Tensor, action: torch.Tensor, 
                     reward: torch.Tensor, next_state: torch.Tensor, 
                     done: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        with torch.no_grad():
            next_q = self.target_network(self.state_encoder(next_state))
            next_v = next_q.max(dim=1)[0]
            target_q = reward + gamma * next_v * (1 - done)
            
        current_q = self.forward(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        return target_q - current_q


class TemporalSARSA(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = TemporalStateEncoder(state_dim, hidden_dim)

        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_features = self.state_encoder(state)
        action_probs = self.policy_net(state_features)
        return action_probs, state_features
    
    def get_td_error(self, state: torch.Tensor, action: torch.Tensor,
                     reward: torch.Tensor, next_state: torch.Tensor,
                     next_action: torch.Tensor, done: torch.Tensor,
                     gamma: float = 0.99) -> torch.Tensor:
        state_features = self.state_encoder(state)
        next_state_features = self.state_encoder(next_state)
        
        current_q = self.q_network(
            torch.cat([state_features, action], dim=-1)
        ).squeeze(-1)
        
        with torch.no_grad():
            next_q = self.q_network(
                torch.cat([next_state_features, next_action], dim=-1)
            ).squeeze(-1)
            target_q = reward + gamma * next_q * (1 - done)
            
        return target_q - current_q 
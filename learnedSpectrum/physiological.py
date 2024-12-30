import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class PhysiologicalStateTracker(nn.Module):
    def __init__(self, input_dim: int, num_states: int = 8):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.state_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_states),
            nn.Softmax(dim=-1)
        )
        
        self.transition_predictor = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Linear(128, num_states)
        )
        
    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None):
        # Encode current physiological state
        state_features = self.state_encoder(x)
        state_probs = self.state_classifier(state_features)
        
        # Predict state transitions if previous state available
        if prev_state is not None:
            prev_features = self.state_encoder(prev_state)
            combined = torch.cat([prev_features, state_features], dim=-1)
            transition_logits = self.transition_predictor(combined)
            return state_probs, transition_logits
            
        return state_probs, None 
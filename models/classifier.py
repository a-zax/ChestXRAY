import torch
import torch.nn as nn

class MIXMLP(nn.Module):
    """
    Module 2: Knowledge-Enhanced Classification (MIX-MLP)
    Dual-path disease classifier with modeled disease correlations.
    """
    def __init__(self, in_dim=512, num_diseases=14, dropout=0.1):
        super().__init__()
        
        # Residual path (Identity-like but with projection)
        self.residual = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_diseases)
        )
        
        # Expansion path (MLP style)
        self.expansion = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, num_diseases)
        )
        
        # Disease correlation matrix (Learnable)
        # Initialized close to identity but with some noise to learn co-occurrences
        self.correlation = nn.Parameter(
            torch.eye(num_diseases) * 0.9 + torch.randn(num_diseases, num_diseases) * 0.01
        )
        
    def forward(self, visual_feat):
        """
        Args:
            visual_feat: [Batch, D] (Typically Organ-level features)
        Returns:
            logits: [Batch, num_diseases]
        """
        # Two paths
        res_out = self.residual(visual_feat)
        exp_out = self.expansion(visual_feat)
        
        # Combine
        logits = res_out + exp_out
        
        # Apply disease correlation
        # shape: [B, num_diseases] @ [num_diseases, num_diseases]
        logits = torch.matmul(logits, self.correlation)
        
        return logits

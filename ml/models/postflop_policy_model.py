# ml/models/postflop_policy_model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostflopPolicyModel(nn.Module):
    """
    Simple MLP policy network for postflop decisions.

    Inputs:
      - Dense numeric features (already encoded)
    Outputs:
      - Action logits (softmax applied in loss / inference)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            logits: [batch_size, output_dim]
        """
        h = self.backbone(x)
        logits = self.policy_head(h)
        return logits
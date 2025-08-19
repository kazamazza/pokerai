from typing import Dict

import torch
from torch import nn


class RangeNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden: int = 256,
                 dropout: float = 0.1,
                 num_actions: int | None = None,
                 output_dim: int | None = None):  # alias for consistency
        super().__init__()
        if num_actions is None and output_dim is None:
            raise ValueError("Provide num_actions or output_dim")
        self.input_dim = input_dim
        self.num_actions = num_actions if num_actions is not None else int(output_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_actions),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x_vec"]
        logits = self.net(x)
        return {"logits": logits}
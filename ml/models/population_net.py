import torch
from torch import nn


class PopulationNet(nn.Module):
    """
    Simple, strong baseline for population profiling.
    - Input: dense feature vector x_vec ∈ R^D (already normalized in ETL)
    - Output: M targets in [0,1] (frequencies / propensities), via sigmoid
    - Optional: pass dropout for regularization
    """
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden // 2, output_dim)

    def forward(self, batch):
        """
        Expect batch["x_vec"] : FloatTensor [B, D]
        Returns predictions in [0,1] : FloatTensor [B, M]
        """
        x = batch["x_vec"]              # [B, D]
        h = self.backbone(x)            # [B, H/2]
        logits = self.head(h)           # [B, M]
        probs = torch.sigmoid(logits)   # constrain to [0,1]
        return probs
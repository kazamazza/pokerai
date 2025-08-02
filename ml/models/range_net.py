import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeNet(nn.Module):
    """
    A simple feedforward neural network for predicting action probabilities
    from structured poker feature vectors.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, output_dim: int = 3):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each layer.
            output_dim (int): Number of output actions (e.g., [FOLD, CALL, RAISE]).
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        # Optional: normalize logits before softmax
        self.logits_bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Action probabilities (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return F.softmax(self.logits_bn(logits), dim=-1)
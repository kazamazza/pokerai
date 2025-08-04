import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from features.stake_levels import STAKES
from features.types import POSITIONS, ACTION_CONTEXTS
from ml.schema.population_net_schema import PopulationNetFeatures, PopulationNetLabel
from utils.encoders import one_hot_encode


class PopulationNetDataset(Dataset):
    """
    PyTorch dataset for training PopulationNet on population-level statistics.
    Returns (input_tensor, label_tensor) for each sample.
    """
    def __init__(self, features: List[PopulationNetFeatures], labels: List[PopulationNetLabel]):
        assert len(features) == len(labels), "Features and labels must be the same length"
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        label = self.labels[idx]

        x = torch.tensor(
            one_hot_encode(feat.stake_level, STAKES) +
            one_hot_encode(feat.action_context, ACTION_CONTEXTS) +
            one_hot_encode(feat.position, POSITIONS) +
            [feat.player_count / 6.0],  # normalize to [0.33–1.0]
            dtype=torch.float32
        )

        y = torch.tensor([
            label.fold_pct,
            label.call_pct,
            label.raise_pct,
            label.avg_bet_sizing
        ], dtype=torch.float32)

        return x, y
from torch.utils.data import Dataset
from typing import List, Tuple

from ml.schema.equity_net_schema import EquityNetFeatures, EquityNetLabel


class EquityNetDataset(Dataset):
    def __init__(self, features: List[EquityNetFeatures], labels: List[EquityNetLabel]):
        assert len(features) == len(labels), "Features and labels must be the same length"
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[EquityNetFeatures, EquityNetLabel]:
        return self.features[idx], self.labels[idx]
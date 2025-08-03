from torch.utils.data import Dataset
from typing import List, Tuple

from ml.schema.population_net_schema import PopulationNetFeatures, PopulationNetLabel


class PopulationNetDataset(Dataset):
    """
    A PyTorch dataset for training the PopulationNet model on aggregated hand history statistics.
    """
    def __init__(self, features: List[PopulationNetFeatures], labels: List[PopulationNetLabel]):
        assert len(features) == len(labels), "Features and labels must be the same length"
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[PopulationNetFeatures, PopulationNetLabel]:
        return self.features[idx], self.labels[idx]
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from ml.schema.equity_net_schema import EquityNetFeatures, EquityNetLabel
from utils.encoders import encode_card, encode_board, encode_position


class EquityNetDataset(Dataset):
    def __init__(self, features: List[EquityNetFeatures], labels: List[EquityNetLabel]):
        assert len(features) == len(labels), "Features and labels must be the same length"
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        label = self.labels[idx]

        x = torch.tensor([
            encode_card(feat.hero_hand[0]),
            encode_card(feat.hero_hand[1]),
            encode_board(feat.board),
            encode_position(feat.position),
            feat.stack_bb / 300.0,
            feat.pot_size / 100.0,
            feat.num_players / 6.0,
            int(feat.has_initiative)
        ], dtype=torch.float32)

        y = torch.tensor([
            label.normalized_equity
        ], dtype=torch.float32)

        return x, y
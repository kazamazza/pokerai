import json, gzip, torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any, List, Union, Tuple
from ml.schema.equity_net_schema import EquityNetFeatures, EquityNetLabel

# Fixed pads so DataLoader can batch without custom collate:
OPP_EMB_DIM     = 32   # max opp emb length you'll ever emit (e.g., PCA-16 → set 16 or keep 32 headroom)
BOARD_FEATS_DIM = 8    # max number of board descriptors you emit

JsonLike = Dict[str, Any]

class EquityNetDataset(Dataset):
    """
    Dataset that reads JSONL(.gz) rows, parses into EquityNetFeatures + EquityNetLabel,
    and emits (X, y) tensors ready for training.
    """
    def __init__(self, path: Union[str, Path]):
        self.samples: List[Tuple[EquityNetFeatures, EquityNetLabel]] = []
        path = str(path)
        open_fn = gzip.open if path.endswith(".gz") else open

        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                feat = EquityNetFeatures(**raw["x"])   # validate & coerce
                lab  = EquityNetLabel(**raw["y"])      # validate & coerce
                self.samples.append((feat, lab))

        if not self.samples:
            raise ValueError(f"No rows found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        feat, lab = self.samples[i]

        # --- categorical
        hand_id = torch.tensor(int(feat.hand_id), dtype=torch.long)

        # board_cluster_id is already validated by Pydantic (>=0)
        board_cluster_id = torch.tensor(int(feat.board_cluster_id), dtype=torch.long)

        # --- opponent representation
        # prefer embedding if present, else bucket_id
        OPP_EMB_DIM = 32  # keep in one place (or import from config)
        opp_emb = torch.zeros(OPP_EMB_DIM, dtype=torch.float32)
        bucket_id = -1
        if feat.opp_range_emb is not None:
            oe = torch.tensor(feat.opp_range_emb, dtype=torch.float32)
            d = min(oe.numel(), OPP_EMB_DIM)
            opp_emb[:d] = oe[:d]
        elif feat.opp_range_bucket_id is not None:
            bucket_id = int(feat.opp_range_bucket_id)
        bucket_id = torch.tensor(bucket_id, dtype=torch.long)

        # --- optional board features
        BOARD_FEATS_DIM = 8
        board_feats = torch.zeros(BOARD_FEATS_DIM, dtype=torch.float32)
        if feat.board_feats is not None:
            bf = torch.tensor(feat.board_feats, dtype=torch.float32)
            d = min(bf.numel(), BOARD_FEATS_DIM)
            board_feats[:d] = bf[:d]

        # --- label
        equity = torch.tensor([float(lab.equity)], dtype=torch.float32)

        X = {
            "hand_id": hand_id,
            "board_cluster_id": board_cluster_id,
            "bucket_id": bucket_id,
            "opp_emb": opp_emb,
            "board_feats": board_feats,
        }
        return X, equity


def load_equity_dataset(path: Union[str, Path]) -> EquityNetDataset:
    return EquityNetDataset(path)
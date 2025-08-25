# ml/datasets/rangenet_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HAND_COUNT = 169

@dataclass
class CardsInfo:
    cards: Dict[str, int]  # cardinalities for each feature (after ID encoding)

class RangeNetDatasetParquet(Dataset):
    """
    Generic RangeNet dataset (works for preflop or postflop) from a Parquet.

    Expect parquet columns like:
      X: e.g. preflop:  ["stack_bb","hero_pos","opener_pos","opener_action"]
         postflop:     ["stack_bb","hero_pos","villain_pos","street","board_cluster_id"]
      Y: y_0 .. y_168   (soft distribution over 169 preflop hand classes)
      W: weight

    You pass:
      - x_cols: list[str] of feature column names to encode as categorical IDs
      - weight_col: name of weight column
    """
    def __init__(
        self,
        parquet_path: str | Path,
        x_cols: Sequence[str],
        weight_col: str = "weight",
        device: Optional[torch.device] = None,
        min_weight: Optional[float] = None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.x_cols = list(x_cols)
        self.weight_col = weight_col
        self.device = device

        df = pd.read_parquet(self.parquet_path)

        # sanity: need y_0..y_168
        y_cols = [f"y_{i}" for i in range(HAND_COUNT)]
        missing = [c for c in (self.x_cols + y_cols + [self.weight_col]) if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        if min_weight is not None:
            df = df[df[self.weight_col] >= float(min_weight)].reset_index(drop=True)

        # --- build encoders for each X col ---
        self._encoders: Dict[str, Dict[Any, int]] = {}
        self._cards: Dict[str, int] = {}
        X_ids = np.zeros((len(df), len(self.x_cols)), dtype=np.int64)
        for j, col in enumerate(self.x_cols):
            uniques = df[col].dropna().unique().tolist()
            try:
                uniques = sorted(uniques)
            except Exception:
                pass
            enc = {v: i for i, v in enumerate(uniques)}
            self._encoders[col] = enc
            self._cards[col] = len(enc)
            # map (unknowns → extra bucket)
            ids = df[col].map(enc).fillna(-1).astype("int64").to_numpy()
            if (ids == -1).any():
                miss_mask = (ids == -1)
                new_id = self._cards[col]
                ids[miss_mask] = new_id
                self._cards[col] = new_id + 1
            X_ids[:, j] = ids

        Y = df[y_cols].to_numpy(dtype="float32")  # (N,169)
        # normalize softly
        s = Y.sum(axis=1, keepdims=True)
        s[s <= 1e-12] = 1.0
        Y = Y / s

        W = df[self.weight_col].to_numpy(dtype="float32")

        self._X = X_ids
        self._Y = Y
        self._W = W
        self.feature_order = list(self.x_cols)
        self.cards_info = CardsInfo(cards=dict(self._cards))
        # keep original df around only if you need stratified splits elsewhere
        self.df = df

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int):
        row = self._X[idx]
        x_dict = {
            self.feature_order[j]: torch.tensor(row[j], dtype=torch.long, device=self.device)
            for j in range(len(self.feature_order))
        }
        y = torch.tensor(self._Y[idx], dtype=torch.float32, device=self.device)
        w = torch.tensor(self._W[idx], dtype=torch.float32, device=self.device)
        return x_dict, y, w

    def cards(self) -> Dict[str, int]:
        return dict(self._cards)

    def id_maps(self) -> Dict[str, Dict[Any, int]]:
        return {k: dict(v) for k, v in self._encoders.items()}

def rangenet_collate_fn(batch):
    keys = batch[0][0].keys()
    x = {k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys}
    y = torch.stack([b[1] for b in batch], dim=0)  # [B,169]
    w = torch.stack([b[2] for b in batch], dim=0)  # [B]
    return x, y, w
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class CardsInfo:
    """Cardinalities for each feature (after ID encoding)."""
    cards: Dict[str, int]

class EquityDatasetParquet(Dataset):
    """
    Generic EquityNet dataset (works for preflop or postflop) from a Parquet.

    Expects config to provide:
      - x_cols:   list of feature column names (e.g. preflop:
                   ["stack_bb","hero_pos","opener_action","hand_id"]
                   postflop adds: "board_cluster_id")
      - y_cols:   ["p_win","p_tie","p_lose"]
      - weight_col: "weight"

    Returns per item:
      x_dict: {col_name: tensor[int]}   # integer IDs per feature
      y:     tensor[3]                  # (p_win, p_tie, p_lose)
      w:     tensor[]                   # scalar weight
    """

    def __init__(
        self,
        parquet_path: str | Path,
        x_cols: Sequence[str],
        y_cols: Sequence[str],
        weight_col: str,
        # Optional row filters
        keep_values: Optional[Dict[str, Sequence[Any]]] = None,
        min_weight: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.x_cols = list(x_cols)
        self.y_cols = list(y_cols)
        self.weight_col = weight_col
        self.device = device

        df = pd.read_parquet(self.parquet_path)

        # --- schema checks ---
        need = set(self.x_cols) | set(self.y_cols) | {self.weight_col}
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"Parquet missing required columns: {miss}")

        # --- optional filtering (pre-encoding) ---
        if keep_values:
            for k, vals in keep_values.items():
                if k not in df.columns:  # silently skip unknown filter keys
                    continue
                df = df[df[k].isin(list(vals))]

        if min_weight is not None:
            df = df[df[self.weight_col] >= float(min_weight)]

        df = df.reset_index(drop=True)

        # --- build per-column ID encoders for X ---
        # We treat every X column as categorical; if it's already int-like,
        # this still yields a dense 0..C-1 ID space for embeddings.
        self._encoders: Dict[str, Dict[Any, int]] = {}
        self._cards: Dict[str, int] = {}

        for col in self.x_cols:
            uniques = df[col].dropna().unique().tolist()
            # stable order: sort stringy cols lexicographically; else numeric
            try:
                uniques = sorted(uniques)
            except Exception:
                pass
            enc = {v: i for i, v in enumerate(uniques)}
            self._encoders[col] = enc
            self._cards[col] = len(enc)

        # --- materialize X, Y, W as numpy for fast indexing ---
        X_ids = np.zeros((len(df), len(self.x_cols)), dtype=np.int64)
        for j, col in enumerate(self.x_cols):
            enc = self._encoders[col]
            # map unknowns (NaN/missing) to -1 then clip to 0 if needed
            X_ids[:, j] = df[col].map(enc).fillna(-1).astype("int64").to_numpy()
            if (X_ids[:, j] == -1).any():
                # if there are unseen/missing, add an extra bucket for them
                miss_mask = (X_ids[:, j] == -1)
                new_id = self._cards[col]
                X_ids[miss_mask, j] = new_id
                self._cards[col] = new_id + 1  # increment cardinality

        Y = df[self.y_cols].to_numpy(dtype="float32")  # (N,3)
        # Normalize softly to be safe
        row_sums = Y.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 1e-12] = 1.0
        Y = Y / row_sums

        W = df[self.weight_col].to_numpy(dtype="float32")  # (N,)

        self._X = X_ids
        self._Y = Y
        self._W = W

        self.feature_order = list(self.x_cols)
        self.cards_info = CardsInfo(cards=dict(self._cards))

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int):
        row = self._X[idx]
        x_dict = {
            # keep names *exactly* as in x_cols so feature_order matches
            self.feature_order[j]: torch.tensor(row[j], dtype=torch.long, device=self.device)
            for j in range(len(self.feature_order))
        }
        y = torch.tensor(self._Y[idx], dtype=torch.float32, device=self.device)
        w = torch.tensor(self._W[idx], dtype=torch.float32, device=self.device)
        return x_dict, y, w

    # helpers to wire into the model init
    def cards(self) -> Dict[str, int]:
        return dict(self._cards)

    def id_maps(self) -> Dict[str, Dict[Any, int]]:
        return {k: dict(v) for k, v in self._encoders.items()}
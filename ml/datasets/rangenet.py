from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.models.constants import HAND_COUNT


# Reuse your existing generic bits

@dataclass
class CardsInfo:
    cards: Dict[str, int]

# --- helpers: canon maps for safety (adjust if your pipeline already guarantees canon) ---
_POS_CANON = {"UTG","HJ","CO","BTN","SB","BB"}
_POS_ALIAS = {"BU":"BTN","MP":"HJ","EP":"UTG","LJ":"HJ"}

def canon_pos(p: Any) -> Optional[str]:
    if not isinstance(p, str): return None
    p = p.strip().upper()
    p = _POS_ALIAS.get(p, p)
    return p if p in _POS_CANON else None

# Mapping of raw street labels to normalized IDs
# Convention: FLOP=1, TURN=2, RIVER=3
_STREET_MAP = {
    1: 1, "1": 1, "FLOP": 1, "flop": 1,
    2: 2, "2": 2, "TURN": 2, "turn": 2,
    3: 3, "3": 3, "RIVER": 3, "river": 3,
}

def canon_street(x: Union[int, str]) -> Optional[int]:
    """
    Normalize street indicator into {1,2,3}.
      - FLOP → 1
      - TURN → 2
      - RIVER → 3
    Returns None if the value cannot be mapped.
    """
    return _STREET_MAP.get(x, None)

# Preflop action canon (keep small and practical)
_ACTION_CANON = {
    "MIN":"RAISE", "RAISE":"RAISE", "OPEN":"RAISE",
    "AI":"ALL_IN", "ALL_IN":"ALL_IN", "SHOVE":"ALL_IN",
    "LIMP":"LIMP", "CALL":"CALL", "FOLD":"FOLD",
    "3BET":"3BET", "4BET":"4BET", "5BET":"5BET"
}

def canon_action(a: Any) -> Optional[str]:
    if not isinstance(a, str): return None
    a = a.strip().upper()
    # accept percent like "60%" as raise
    if a.endswith("%") and a[:-1].isdigit():
        return "RAISE"
    return _ACTION_CANON.get(a, a)

# Your Ctx enum values – ensure parquet stores ints (recommended) or exact strings you map here
_CTX_MAP = {
    "OPEN": 0, "VS_OPEN": 1, "VS_3BET": 2, "VS_4BET": 3,
    "BLIND_VS_STEAL": 4, "LIMPED_SINGLE": 5, "LIMPED_MULTI": 6,
    "VS_CBET": 10, "VS_CBET_TURN": 11, "VS_CHECK_RAISE": 13, "VS_DONK": 14,
}

def canon_ctx(v: Any) -> Optional[int]:
    if v is None: return None
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, str):
        s = v.strip().upper()
        if s in _CTX_MAP: return _CTX_MAP[s]
        # allow numeric-in-string
        if s.isdigit(): return int(s)
    return None


class RangeNetDatasetParquet(Dataset):
    """
    Generic RangeNet dataset (works for preflop or postflop) from a Parquet.

    X: categorical features → ID-encoded
    Y: y_0..y_168 (soft 169-vector)
    W: weight column

    Use this directly, or via PreflopRangeDatasetParquet for locked schema.
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

        y_cols = [f"y_{i}" for i in range(HAND_COUNT)]
        missing = [c for c in (self.x_cols + y_cols + [self.weight_col]) if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        if min_weight is not None:
            df = df[df[self.weight_col] >= float(min_weight)].reset_index(drop=True)

        # build per-column encoders
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
            ids = df[col].map(enc).fillna(-1).astype("int64").to_numpy()
            if (ids == -1).any():
                miss_mask = (ids == -1)
                new_id = self._cards[col]
                ids[miss_mask] = new_id
                self._cards[col] = new_id + 1
            X_ids[:, j] = ids

        Y = df[y_cols].to_numpy(dtype="float32")
        s = Y.sum(axis=1, keepdims=True)
        s[s <= 1e-12] = 1.0
        Y = Y / s

        W = df[self.weight_col].to_numpy(dtype="float32")

        self._X = X_ids
        self._Y = Y
        self._W = W
        self.feature_order = list(self.x_cols)
        self.cards_info = CardsInfo(cards=dict(self._cards))
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
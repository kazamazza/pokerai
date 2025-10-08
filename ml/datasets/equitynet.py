from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class CardsInfo:
    cards: Dict[str, int]

class EquityDatasetParquet(Dataset):

    def __init__(
        self,
        parquet_path: str | Path,
        x_cols: Sequence[str],
        y_cols: Sequence[str],
        weight_col: str,
        keep_values: Optional[Dict[str, Sequence[Any]]] = None,
        min_weight: Optional[float] = None,
        device: Optional[torch.device] = None,
        cont_cols: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.x_cols = list(x_cols)
        self.y_cols = list(y_cols)
        self.cont_cols = list(cont_cols or [])
        self.weight_col = weight_col
        self.device = device

        df = pd.read_parquet(self.parquet_path)

        need = set(self.x_cols) | set(self.y_cols) | {self.weight_col}
        need |= {c for c in self.cont_cols}
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"Parquet missing required columns: {miss}")

        if keep_values:
            for k, vals in keep_values.items():
                if k not in df.columns:  # silently skip unknown filter keys
                    continue
                df = df[df[k].isin(list(vals))]

        if min_weight is not None:
            df = df[df[self.weight_col] >= float(min_weight)]

        df = df.reset_index(drop=True)

        self._encoders: Dict[str, Dict[Any, int]] = {}
        self._cards: Dict[str, int] = {}

        for col in self.x_cols:
            uniques = df[col].dropna().unique().tolist()
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

        self._cont_arrays: Dict[str, np.ndarray] = {}
        for c in self.cont_cols:
            if c == "board_mask_52":
                # Expect each cell to be a list/array of length 52
                arr = np.stack(
                    df[c].map(lambda v: np.asarray(v if v is not None else [0.0] * 52, dtype=np.float32)).to_list(),
                    axis=0
                )  # shape (N,52)
                if arr.shape[1] != 52:
                    raise ValueError(f"board_mask_52 must be shape (N,52), got {arr.shape}")
                self._cont_arrays[c] = arr
            else:
                # scalars -> (N,1)
                self._cont_arrays[c] = df[c].astype(np.float32).to_numpy().reshape(-1, 1)

        self.feature_order = list(self.x_cols)
        self.cards_info = CardsInfo(cards=dict(self._cards))
        self.df = df.copy()

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int):
        row = self._X[idx]
        x_dict = {
            self.feature_order[j]: torch.tensor(row[j], dtype=torch.long, device=self.device)
            for j in range(len(self.feature_order))
        }
        # continuous features dict
        c_dict: Dict[str, torch.Tensor] = {}
        for k, arr in self._cont_arrays.items():
            c_dict[k] = torch.tensor(arr[idx], dtype=torch.float32, device=self.device)

        y = torch.tensor(self._Y[idx], dtype=torch.float32, device=self.device)
        w = torch.tensor(self._W[idx], dtype=torch.float32, device=self.device)
        return x_dict, c_dict, y, w

    # helpers to wire into the model init
    def cards(self) -> Dict[str, int]:
        return dict(self._cards)

    def id_maps(self) -> Dict[str, Dict[Any, int]]:
        return {k: dict(v) for k, v in self._encoders.items()}

def equity_collate_fn(batch):
    # batch[i] = (x_dict, c_dict, y, w)
    x_keys = batch[0][0].keys()
    c_keys = batch[0][1].keys()

    x_dict = {k: torch.stack([item[0][k].long()  for item in batch], dim=0) for k in x_keys}
    c_dict = {k: torch.stack([item[1][k].float() for item in batch], dim=0) for k in c_keys}

    y = torch.stack([item[2].float() for item in batch], dim=0)
    w = torch.stack([item[3].float() for item in batch], dim=0)
    return x_dict, c_dict, y, w
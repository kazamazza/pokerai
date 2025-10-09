from __future__ import annotations
from typing import Sequence, Optional, Iterable, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --- Stable defaults matching the parquet builder ---
POP_X_COLS        = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "spr_bin", "bet_size_bucket"]
POP_Y_COLS_SOFT   = ["p_fold", "p_call", "p_raise"]
POP_Y_COL_HARD    = "y"
POP_W_COL         = "w"

FeatureVector = torch.LongTensor
LabelVector   = torch.FloatTensor


class PopulationDatasetParquet(Dataset):
    """
    Dataset over aggregated 'cells' for PopulationNet.

    One row (cell) corresponds to a unique combination of:
      [stakes_id, street_id, ctx_id, hero_pos_id, villain_pos_id]

    Returns a tuple (x_dict, y, w):
      - x_dict: {feature_name: LongTensor[]}   integer IDs for categorical features
      - y:      FloatTensor[3]  (soft)  OR  LongTensor[] (hard class index)
      - w:      FloatTensor[]   per-sample weight
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        x_cols: Sequence[str] = POP_X_COLS,
        use_soft_labels: bool = True,                 # <-- choose soft (default) or hard
        y_cols_soft: Sequence[str] = POP_Y_COLS_SOFT,
        y_col_hard: str = POP_Y_COL_HARD,
        weight_col: str = POP_W_COL,
        # Optional filters:
        keep_ctx_ids: Optional[Iterable[int]] = None,
        keep_street_ids: Optional[Iterable[int]] = None,
        min_weight: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.x_cols       = tuple(x_cols)
        self.use_soft_labels = bool(use_soft_labels)
        self.y_cols_soft  = tuple(y_cols_soft)
        self.y_col_hard   = y_col_hard
        self.weight_col   = weight_col
        self.device       = device

        # --- Load parquet ---
        df = pd.read_parquet(self.parquet_path)

        # --- Required columns check ---
        required = set(x_cols) | {weight_col}
        required |= set(y_cols_soft) if use_soft_labels else {y_col_hard}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        # ensure categorical columns are integers (builder already emits ints; this is just guardrail)
        for c in x_cols:
            if not np.issubdtype(df[c].dtype, np.integer):
                df[c] = df[c].astype("int64")

        # --- Optional filtering ---
        if keep_ctx_ids is not None:
            df = df[df["ctx_id"].isin(list(keep_ctx_ids))]
        if keep_street_ids is not None:
            df = df[df["street_id"].isin(list(keep_street_ids))]
        if min_weight is not None:
            df = df[df[self.weight_col] >= float(min_weight)]

        df = df.reset_index(drop=True)
        self._df = df  # keep around for introspection

        # --- Pre-extract arrays for speed ---
        self._X_np = df[list(self.x_cols)].to_numpy(dtype="int64")

        if self.use_soft_labels:
            Y = df[list(self.y_cols_soft)].to_numpy(dtype="float32")
            # Defensive normalization to ensure rows sum to 1
            row_sums = Y.sum(axis=1, keepdims=True)
            # fallback for degenerate rows (shouldn't happen if n_rows>0)
            bad = (row_sums <= 1e-8)
            if bad.any():
                Y[bad.squeeze()] = np.array([1.0, 0.0, 0.0], dtype="float32")
                row_sums = Y.sum(axis=1, keepdims=True)
            self._Y_np = (Y / row_sums).astype("float32")  # shape [N, 3]
        else:
            self._Y_np = df[self.y_col_hard].to_numpy(dtype="int64")   # shape [N]

        self._W_np = df[self.weight_col].to_numpy(dtype="float32")     # shape [N]

    def __len__(self) -> int:
        return self._X_np.shape[0]

    def __getitem__(self, idx: int):
        # X as dict of per-feature scalar tensors (default collate or our collate stacks them to [B])
        row = self._X_np[idx]
        x_dict = {name: torch.tensor(row[i], dtype=torch.long, device=self.device)
                  for i, name in enumerate(self.x_cols)}

        if self.use_soft_labels:
            y = torch.tensor(self._Y_np[idx], dtype=torch.float32, device=self.device)  # [3]
        else:
            y = torch.tensor(self._Y_np[idx], dtype=torch.long, device=self.device)     # scalar class

        w = torch.tensor(self._W_np[idx], dtype=torch.float32, device=self.device)      # scalar
        return x_dict, y, w

    # Convenience access to the underlying DataFrame
    @property
    def df(self) -> pd.DataFrame:
        return self._df


    def __repr__(self) -> str:
        mode = "soft" if self.use_soft_labels else "hard"
        return (f"PopulationDatasetParquet(n={len(self)}, mode={mode}, "
                f"x_cols={list(self.x_cols)}, "
                f"y_cols={list(self.y_cols_soft) if self.use_soft_labels else self.y_col_hard}, "
                f"w_col={self.weight_col})")


def population_collate_fn(batch):
    """
    Robust collate: stacks dict-of-scalars into dict-of-[B] tensors,
    and stacks y/w appropriately regardless of hard/soft mode.
    """
    xs, ys, ws = zip(*batch)  # list of dicts, list of tensors, list of tensors

    # Stack features by key
    x_keys = xs[0].keys()
    x_batch = {k: torch.stack([x[k] for x in xs], dim=0) for k in x_keys}  # each is [B]

    # Stack labels: soft => [B, 3], hard => [B]
    y0 = ys[0]
    if y0.dim() == 1:  # soft row is shape [3]
        y_batch = torch.stack(ys, dim=0)  # [B, 3]
    else:              # hard scalar (0-d)
        y_batch = torch.stack(ys, dim=0)  # [B]

    # Stack weights -> [B]
    w_batch = torch.stack(ws, dim=0)

    return x_batch, y_batch, w_batch
# ml/data/population_dataset.py
#
# A tiny, readable Dataset for PopulationNet.
# One row == one "cell" (a specific poker situation).
#
# Expected Parquet columns (one row per cell):
#   - stakes_id:       int
#   - street_id:       int        (0: PREFLOP, 1: FLOP, 2: TURN, 3: RIVER)
#   - ctx_id:          int        (e.g., VS_OPEN, VS_3BET, VS_CBET, etc., via your Ctx enum)
#   - hero_pos_id:     int        (0..5 for 6-max mapping)
#   - villain_pos_id:  int
#   - y_fold:          float      (target prob)
#   - y_call:          float      (target prob)
#   - y_raise:         float      (target prob)
#   - weight:          float      (confidence, usually clipped n_rows)
#   - n_rows:          int        (original evidence count; not used in training directly)
#
# Returns:
#   X: LongTensor[int]    shape (num_features,)  -> categorical ids
#   Y: FloatTensor[float] shape (3,)             -> [p_fold, p_call, p_raise]
#   W: FloatTensor[float] shape ()               -> scalar weight
#
# Notes:
# - Keep features as raw integer IDs (we’ll embed them in the model).
# - We lightly validate/repair Y to ensure it sums ~1.
# - You can filter by allowed contexts/streets/positions with simple args.

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset


# --- Stable constants for PopulationNet dataset ---
FeatureVector = torch.LongTensor
LabelVector = torch.FloatTensor

POP_X_COLS = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"]
POP_Y_COLS = ["y_fold", "y_call", "y_raise"]
POP_W_COL  = "weight"


class PopulationDatasetParquet(Dataset):
    """
    PopulationNet dataset from a Parquet of aggregated 'cells'.

    Each row = one cell (context/position/street/stake).

    One item is (X, Y, W):
      X = [stakes_id, street_id, ctx_id, hero_pos_id, villain_pos_id]  (int ids)
      Y = [p_fold, p_call, p_raise]                                    (soft label probs)
      W = scalar weight (float), e.g. clipped n_rows
    """

    def __init__(
        self,
        parquet_path: str | Path,
        x_cols: Sequence[str] = POP_X_COLS,
        y_cols: Sequence[str] = POP_Y_COLS,
        weight_col: str = POP_W_COL,
        # Optional filters
        keep_ctx_ids: Optional[Iterable[int]] = None,
        keep_street_ids: Optional[Iterable[int]] = None,
        min_weight: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.x_cols = tuple(x_cols)
        self.y_cols = tuple(y_cols)
        self.weight_col = weight_col
        self.device = device

        df = pd.read_parquet(self.parquet_path)

        # --- Basic required columns check ---
        required = set(self.x_cols) | set(self.y_cols) | {self.weight_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        # --- Optional filtering ---
        if keep_ctx_ids is not None:
            df = df[df["ctx_id"].isin(list(keep_ctx_ids))]
        if keep_street_ids is not None:
            df = df[df["street_id"].isin(list(keep_street_ids))]
        if min_weight is not None:
            df = df[df[self.weight_col] >= float(min_weight)]

        # Reset index after filtering
        df = df.reset_index(drop=True)

        # Store in memory
        self._df = df

        # Pre-extract numpy arrays
        self._X_np = df[list(self.x_cols)].to_numpy(dtype="int64")
        self._Y_np = df[list(self.y_cols)].to_numpy(dtype="float32")
        self._W_np = df[self.weight_col].to_numpy(dtype="float32")

        # --- Clean up Y to ensure valid distributions ---
        row_sums = self._Y_np.sum(axis=1, keepdims=True)
        zero_mask = (row_sums.squeeze() <= 1e-8)
        if zero_mask.any():
            # Neutral fallback if no evidence: [1, 0, 0] (fold always)
            self._Y_np[zero_mask] = [1.0, 0.0, 0.0]
            row_sums = self._Y_np.sum(axis=1, keepdims=True)
        self._Y_np = self._Y_np / row_sums

    def __len__(self) -> int:
        return self._X_np.shape[0]

    def __getitem__(self, idx: int) -> Tuple[FeatureVector, LabelVector, torch.Tensor]:
        x = torch.as_tensor(self._X_np[idx], dtype=torch.long, device=self.device)
        y = torch.as_tensor(self._Y_np[idx], dtype=torch.float32, device=self.device)
        w = torch.as_tensor(self._W_np[idx], dtype=torch.float32, device=self.device)
        return x, y, w

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# We model 169 distinct preflop hands (AA..72o)
HAND_COUNT = 169

# Feature columns present in the Parquet you built
RAW_COLS = ["stack_bb", "hero_pos", "opener_action", "hand_id", "p_freq", "weight"]

@dataclass
class EquityCards:
    """Cardinalities for categorical features."""
    stack_bb: int
    hero_pos: int
    opener_action: int

class EquityNetParquet(Dataset):
    """
    Dataset for EquityNet (preflop, from Monker manifest -> aggregated Parquet).

    Parquet schema (one row per (scenario, hand_id)):
      - stack_bb:int
      - hero_pos:str
      - opener_action:str
      - hand_id:int (0..168)
      - p_freq:float   (frequency for this hand_id in scenario)
      - weight:float   (row weight, usually 1 per file, later summed to scenario)

    We transform this into one item per *scenario*:
      X = {stack_bb_id, hero_pos_id, opener_action_id}  (categorical IDs)
      Y = float[169] (soft label distribution across hands; sums to 1)
      W = float       (scenario weight = sum(weight) over its 169 rows)
    """

    def __init__(
        self,
        parquet_path: str | Path,
        # Optional filters
        keep_stack_bb: Optional[Iterable[int]] = None,
        keep_hero_pos: Optional[Iterable[str]] = None,
        keep_opener_action: Optional[Iterable[str]] = None,
        min_weight: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.device = device

        df = pd.read_parquet(self.parquet_path)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        # --- Optional filtering on raw rows ---
        if keep_stack_bb is not None:
            keep_stack_bb = set(int(x) for x in keep_stack_bb)
            df = df[df["stack_bb"].isin(keep_stack_bb)]
        if keep_hero_pos is not None:
            keep_hero_pos = set(str(x) for x in keep_hero_pos)
            df = df[df["hero_pos"].isin(keep_hero_pos)]
        if keep_opener_action is not None:
            keep_opener_action = set(str(x) for x in keep_opener_action)
            df = df[df["opener_action"].isin(keep_opener_action)]

        # --- Build categorical id maps (stable, by sorted unique) ---
        stack_uniques = sorted(df["stack_bb"].unique().tolist())
        pos_uniques   = sorted(df["hero_pos"].unique().tolist())
        act_uniques   = sorted(df["opener_action"].unique().tolist())

        self.stack_to_id: Dict[int, int] = {v: i for i, v in enumerate(stack_uniques)}
        self.pos_to_id:   Dict[str, int] = {v: i for i, v in enumerate(pos_uniques)}
        self.act_to_id:   Dict[str, int] = {v: i for i, v in enumerate(act_uniques)}

        self.cards = EquityCards(
            stack_bb=len(self.stack_to_id),
            hero_pos=len(self.pos_to_id),
            opener_action=len(self.act_to_id),
        )

        # --- Pivot to one row per scenario with Y[169] and W ---
        # group key = (stack_bb, hero_pos, opener_action)
        key_cols = ["stack_bb", "hero_pos", "opener_action"]

        # Aggregate scenario weights (sum raw weights across hand rows)
        w_df = df.groupby(key_cols, as_index=False)["weight"].sum().rename(columns={"weight": "W"})

        # Build dense 169-length arrays per scenario
        # Start with zeros and fill by hand_id.
        grouped = df.groupby(key_cols)
        X_list: List[Tuple[int, int, int]] = []
        Y_list: List[np.ndarray] = []
        W_list: List[float] = []

        # Quick lookup of scenario -> W
        w_key = w_df.set_index(key_cols)["W"].to_dict()

        for key, g in grouped:
            stack_bb, hero_pos, opener_action = key
            y = np.zeros(HAND_COUNT, dtype=np.float32)

            # Fill frequencies for present hand_ids
            # Multiple rows per hand_id are possible; sum them.
            # (If you already ensured one row per hand_id, this just works too.)
            sub = g.groupby("hand_id")["p_freq"].sum()
            y[sub.index.values] = sub.values.astype(np.float32)

            # Normalize to sum=1 (avoid zero-division)
            s = y.sum()
            if s <= 1e-12:
                # Fallback to uniform if something is off
                y[:] = 1.0 / HAND_COUNT
            else:
                y /= s

            W = float(w_key.get((stack_bb, hero_pos, opener_action), 0.0))

            # Optional filter by min_weight at scenario level
            if (min_weight is not None) and (W < float(min_weight)):
                continue

            X_list.append((
                self.stack_to_id[int(stack_bb)],
                self.pos_to_id[str(hero_pos)],
                self.act_to_id[str(opener_action)],
            ))
            Y_list.append(y)
            W_list.append(W)

        # Store as numpy for fast indexing
        self._X = np.array(X_list, dtype=np.int64)                 # [N, 3]
        self._Y = np.stack(Y_list, axis=0).astype(np.float32)      # [N, 169]
        self._W = np.array(W_list, dtype=np.float32)               # [N]

        # Names to keep the same order throughout (for the model)
        self.feature_order = ["stack_bb_id", "hero_pos_id", "opener_action_id"]

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int):
        row = self._X[idx]                   # [3]
        x_dict = {
            "stack_bb_id": torch.tensor(row[0], dtype=torch.long, device=self.device),
            "hero_pos_id": torch.tensor(row[1], dtype=torch.long, device=self.device),
            "opener_action_id": torch.tensor(row[2], dtype=torch.long, device=self.device),
        }
        y = torch.tensor(self._Y[idx], dtype=torch.float32, device=self.device)  # [169]
        w = torch.tensor(self._W[idx], dtype=torch.float32, device=self.device)  # []
        return x_dict, y, w

    # Optional helpers
    def cards_dict(self) -> Dict[str, int]:
        return {
            "stack_bb_id": self.cards.stack_bb,
            "hero_pos_id": self.cards.hero_pos,
            "opener_action_id": self.cards.opener_action,
        }

    def id_maps(self) -> Dict[str, Dict]:
        return {
            "stack_bb": self.stack_to_id,
            "hero_pos": self.pos_to_id,
            "opener_action": self.act_to_id,
        }

def equity_collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]):
    # batch is list of (x_dict, y, w)
    # Stack dict features into tensors
    keys = batch[0][0].keys()
    x_dict = {k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys}
    y = torch.stack([b[1] for b in batch], dim=0)  # [B,169]
    w = torch.stack([b[2] for b in batch], dim=0)  # [B]
    return x_dict, y, w
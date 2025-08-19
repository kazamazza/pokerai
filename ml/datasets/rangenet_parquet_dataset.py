# ml/datasets/rangenet_parquet_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------- Hand 169 mapping ----------
_RANKS = "AKQJT98765432"

def _pair(i): return _RANKS[i] + _RANKS[i]
def _suited(i, j):  return _RANKS[i] + _RANKS[j] + "s"   # i < j
def _offsuit(i, j): return _RANKS[i] + _RANKS[j] + "o"   # i > j

def hand169_order() -> List[str]:
    names = []
    for i in range(13):
        for j in range(13):
            if i == j:       names.append(_pair(i))
            elif i < j:      names.append(_suited(i, j))
            else:            names.append(_offsuit(i, j))
    return names

HAND169 = hand169_order()
HAND2IDX: Dict[str, int] = {h: i for i, h in enumerate(HAND169)}

# ---------- Dataset ----------
class RangeNetParquetDataset(Dataset):
    def __init__(
        self,
        path: Path | str,
        ctx_vocab: Optional[List[str]] = None,
        pos_vocab: Optional[List[str]] = None,
        stack_scale: float = 100.0,        # scale stacks to ~[0,1]
        min_actions: int = 2,              # distribution length check
        drop_invalid: bool = True,         # drop rows with malformed targets
    ):
        self.path = Path(path)
        self.df = pd.read_parquet(self.path)

        # ---- Ensure required columns exist ----
        required = ["hero_pos", "ctx", "stack_bb", "hand_bucket", "action_probs"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"{self.path} missing columns: {missing}")

        # ---- Coerce action_probs into list[float] ----
        self.df["action_probs"] = self.df["action_probs"].apply(self._coerce_probs)

        # Optionally drop invalid rows (None or too short)
        if drop_invalid:
            self.df = self.df[self.df["action_probs"].map(lambda v: isinstance(v, list) and len(v) >= min_actions)]

        # ---- Normalize dtypes ----
        self.df["hero_pos"]    = self.df["hero_pos"].astype(str)
        self.df["ctx"]         = self.df["ctx"].astype(str)
        self.df["stack_bb"]    = pd.to_numeric(self.df["stack_bb"], errors="coerce").astype(float)
        self.df["hand_bucket"] = self.df["hand_bucket"].astype(str).str.upper()

        # Drop any remaining NA in the essentials
        self.df = self.df.dropna(subset=["hero_pos", "ctx", "stack_bb", "hand_bucket"])

        # ---- Build / freeze vocabs ----
        self.pos_list = pos_vocab or ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
        self.ctx_list = ctx_vocab or sorted(self.df["ctx"].unique().tolist())
        self.pos2i = {p: i for i, p in enumerate(self.pos_list)}
        self.ctx2i = {c: i for i, c in enumerate(self.ctx_list)}

        # ---- Output dimension from first valid row ----
        first = None
        for v in self.df["action_probs"]:
            if isinstance(v, list) and len(v) >= min_actions:
                first = v
                break
        if first is None:
            raise ValueError("No valid action_probs rows found.")
        self.out_dim = len(first)

        # ---- Input dimension (features): pos(6) + ctx(|C|) + stack(1) + hand169(169)
        self.input_dim = len(self.pos_list) + len(self.ctx_list) + 1 + 169

        # ---- cache ----
        self.stack_scale = float(stack_scale)
        self.N = len(self.df)

    # ---------- utils ----------
    @staticmethod
    def _coerce_probs(v: Any) -> Optional[List[float]]:
        """Handle list, tuple, numpy array, or JSON-encoded strings."""
        if v is None:
            return None
        if isinstance(v, list):
            return [float(x) for x in v]
        if isinstance(v, tuple):
            return [float(x) for x in v]
        if isinstance(v, np.ndarray):
            return [float(x) for x in v.tolist()]
        if isinstance(v, str):
            s = v.strip()
            # Try JSON decode (e.g. "[0.2, 0.8, 0.0]")
            try:
                obj = json.loads(s)
                if isinstance(obj, (list, tuple, np.ndarray)):
                    return [float(x) for x in obj]
            except Exception:
                # Fallback: comma separated "0.2,0.8,0.0"
                try:
                    parts = [p for p in s.split(",") if p]
                    return [float(x) for x in parts]
                except Exception:
                    return None
        # Unsupported type
        try:
            return [float(v)]
        except Exception:
            return None

    def __len__(self) -> int:
        return self.N

    def _vec_row(self, r: pd.Series) -> torch.Tensor:
        # hero_pos one-hot (6)
        pos_oh = torch.zeros(len(self.pos_list), dtype=torch.float32)
        i = self.pos2i.get(r.hero_pos)
        if i is not None:
            pos_oh[i] = 1.0

        # ctx one-hot (|C|)
        ctx_oh = torch.zeros(len(self.ctx_list), dtype=torch.float32)
        j = self.ctx2i.get(r.ctx)
        if j is not None:
            ctx_oh[j] = 1.0

        # stack scaled
        stack = torch.tensor([float(r.stack_bb) / self.stack_scale], dtype=torch.float32)

        # hand 169 one-hot
        hand_oh = torch.zeros(169, dtype=torch.float32)
        hid = HAND2IDX.get(r.hand_bucket)
        if hid is not None:
            hand_oh[hid] = 1.0

        return torch.cat([pos_oh, ctx_oh, stack, hand_oh], dim=0)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        r = self.df.iloc[idx]
        x_vec = self._vec_row(r)

        # Target distribution (normalize; fallback to 'fold' index 0)
        probs = self._coerce_probs(r.action_probs)
        if not probs:
            tgt = torch.zeros(self.out_dim, dtype=torch.float32)
            tgt[0] = 1.0
        else:
            tgt = torch.tensor([float(x) for x in probs], dtype=torch.float32)
            s = float(tgt.sum().item())
            if s <= 0:
                tgt = torch.zeros_like(tgt)
                tgt[0] = 1.0
            else:
                tgt = tgt / s

        return {"x_vec": x_vec}, {"y_dist": tgt}
# ml/etl/ev/ranges.py

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

class VillainRangeProvider:
    """
    Minimal loader for preflop ranges from a parquet with y_0..y_168
    keyed by (hero_pos, villain_pos, stack_bb, action_seq_1..3).
    """
    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)

    @staticmethod
    def _ser_seq(seq: List[str]) -> Tuple[str, str, str]:
        a1 = seq[0] if len(seq) > 0 else ""
        a2 = seq[1] if len(seq) > 1 else ""
        a3 = seq[2] if len(seq) > 2 else ""
        return a1, a2, a3

    def _lookup(self, hero_pos: str, villain_pos: str, stack: float, seq: List[str]) -> Optional[np.ndarray]:
        a1, a2, a3 = self._ser_seq(seq)
        m = (
            (self.df.hero_pos == hero_pos)
            & (self.df.villain_pos == villain_pos)
            & (self.df.stack_bb == float(stack))
            & (self.df.action_seq_1 == a1)
            & (self.df.action_seq_2 == a2)
            & (self.df.action_seq_3 == a3)
        )
        hit = self.df[m]
        if hit.empty:
            return None
        row = hit.iloc[0]
        vec = np.array([row[f"y_{i}"] for i in range(169)], dtype=np.float32)
        s = float(vec.sum())
        return (vec / s) if s > 1e-9 else np.ones(169, dtype=np.float32) / 169.0

    def get_vector(self, hero_pos: str, villain_pos: str, stack: float, action_seq: Optional[List[str]] = None) -> \
    Optional[np.ndarray]:
        seq = action_seq or []
        a1, a2, a3 = self._ser_seq(seq)
        m = (
                (self.df.hero_pos == hero_pos)
                & (self.df.villain_pos == villain_pos)
                & (self.df.stack_bb == float(stack))
                & (self.df.action_seq_1 == a1)
                & (self.df.action_seq_2 == a2)
                & (self.df.action_seq_3 == a3)
        )
        hit = self.df[m]
        if hit.empty:
            return None
        row = hit.iloc[0]
        vec = np.array([row[f"y_{i}"] for i in range(169)], dtype=np.float32)
        s = float(vec.sum())
        return (vec / s) if s > 1e-9 else np.ones(169, dtype=np.float32) / 169.0
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


class VillainRangeProvider:
    def __init__(self, path: str):
        self.df = pd.read_parquet(path)

    @staticmethod
    def _serialize_action_seq(action_seq: List[str]) -> Tuple[str, str, str]:
        a1 = action_seq[0] if len(action_seq) > 0 else ""
        a2 = action_seq[1] if len(action_seq) > 1 else ""
        a3 = action_seq[2] if len(action_seq) > 2 else ""
        return a1, a2, a3

    def _lookup_exact(self, hero_pos, villain_pos, stack, action_seq):
        a1, a2, a3 = self._serialize_action_seq(action_seq)
        df = self.df
        m = (
            (df.hero_pos == str(hero_pos)) &
            (df.villain_pos == str(villain_pos)) &
            (df.stack_bb == float(stack)) &
            (df.action_seq_1 == a1) &
            (df.action_seq_2 == a2) &
            (df.action_seq_3 == a3)
        )
        match = df[m]
        if match.empty:
            raise KeyError("No exact match")
        row = match.iloc[0]
        vec = np.array([row[f"y_{i}"] for i in range(169)], dtype=np.float32)
        s = float(vec.sum())
        return vec / s if s > 1e-9 else np.ones(169, dtype=np.float32) / 169.0

    def _lookup_nearest_stack(self, hero_pos, villain_pos, stack, action_seq):
        a1, a2, a3 = self._serialize_action_seq(action_seq)
        df = self.df
        subset = df[
            (df.hero_pos == str(hero_pos)) &
            (df.villain_pos == str(villain_pos)) &
            (df.action_seq_1 == a1) &
            (df.action_seq_2 == a2) &
            (df.action_seq_3 == a3)
        ]
        if subset.empty:
            raise KeyError("No rows for positions/action_seq")
        # nearest by absolute difference
        svals = subset["stack_bb"].astype(float).values
        idx = int(np.argmin(np.abs(svals - float(stack))))
        row = subset.iloc[idx]
        vec = np.array([row[f"y_{i}"] for i in range(169)], dtype=np.float32)
        s = float(vec.sum())
        return vec / s if s > 1e-9 else np.ones(169, dtype=np.float32) / 169.0

    def get_range_vector(self, hero_pos, villain_pos, stack, action_seq):
        try:
            return self._lookup_exact(hero_pos, villain_pos, stack, action_seq)
        except KeyError:
            pass
        # round stack
        try:
            stack2 = round(float(stack) / 20.0) * 20.0
            return self._lookup_exact(hero_pos, villain_pos, stack2, action_seq)
        except KeyError:
            pass
        # nearest stack
        try:
            return self._lookup_nearest_stack(hero_pos, villain_pos, stack, action_seq)
        except KeyError:
            pass
        # final fallback: empty action seq
        try:
            return self._lookup_nearest_stack(hero_pos, villain_pos, stack, [])
        except KeyError:
            return None
from typing import Sequence, Tuple, Dict

import numpy as np
import pandas as pd


def categorical_cardinalities(df: pd.DataFrame, x_cols: Sequence[str]) -> Dict[str, int]:
    return {col: int(df[col].max()) + 1 for col in x_cols}

def stratified_indices(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("ctx_id", "street_id"),
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = df.groupby(list(group_cols)).indices
    rng = np.random.default_rng(seed)

    train_idx, val_idx = [], []
    for _, idxs in groups.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        split = int(len(idxs) * train_frac)
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])
    return np.array(train_idx), np.array(val_idx)
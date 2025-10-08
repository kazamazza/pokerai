from typing import Sequence, Tuple, Dict

import numpy as np
import pandas as pd


def categorical_cardinalities(df: pd.DataFrame, x_cols: Sequence[str]) -> Dict[str, int]:
    return {col: int(df[col].max()) + 1 for col in x_cols}

def stratified_indices(df: pd.DataFrame, keys, train_frac: float, seed: int):
    import numpy as np
    keys = list(keys) if keys else []
    if not keys or any(k not in df.columns for k in keys):
        n = len(df); idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
        cut = max(1, int(round(n * train_frac)))
        return idx[:cut].tolist(), idx[cut:].tolist()
    strata = df[keys].astype(str).agg("||".join, axis=1).values
    uniq, counts = np.unique(strata, return_counts=True)
    if (counts < 2).any() or len(uniq) > len(df) * 0.8:
        n = len(df); idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
        cut = max(1, int(round(n * train_frac)))
        return idx[:cut].tolist(), idx[cut:].tolist()
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for s in uniq:
        s_idx = np.where(strata == s)[0]
        rng.shuffle(s_idx)
        cut = max(1, min(len(s_idx) - 1, int(round(len(s_idx) * train_frac)))) if len(s_idx) > 1 else 1
        train_idx.extend(s_idx[:cut].tolist())
        val_idx.extend(s_idx[cut:].tolist())
    if not val_idx:
        val_idx = train_idx[-1:]; train_idx = train_idx[:-1]
    return train_idx, val_idx
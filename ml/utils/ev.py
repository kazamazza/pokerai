# file: ml/data/ev_utils.py
from __future__ import annotations

from typing import Any, Iterable, List
import numpy as np

def _board_mask_from_row(row: Any) -> List[float]:
    """Return a 52-length board mask from a parquet/DataFrame row.
    Priority: 'board_mask_52' vector → bm0..bm51 columns → compute from 'board' → zeros."""
    # Helper to access keys in both dict-like and pandas Series
    def _has(k: str) -> bool:
        try:
            if hasattr(row, "keys"):
                return k in row.keys()
            if hasattr(row, "index"):
                return k in row.index
            return k in row  # dict-like
        except Exception:
            return False

    # 1) Direct vector field
    if _has("board_mask_52"):
        v = row["board_mask_52"]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 52:
            return [float(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]

    # 2) Expanded columns bm0..bm51
    keys = [f"bm{i}" for i in range(52)]
    if all(_has(k) for k in keys):
        return [float(row[k]) for k in keys]

    # 3) Compute from raw 'board' string if available
    if _has("board"):
        board = str(row["board"] or "").strip()
        if board:
            try:
                from ml.utils.board_mask import make_board_mask_52  # your existing helper
                v = make_board_mask_52(board)
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 52:
                    return [float(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
            except Exception:
                pass

    # 4) Fallback: zeros
    return [0.0] * 52
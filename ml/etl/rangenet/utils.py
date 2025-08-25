from typing import Dict, List
import numpy as np
import pandas as pd
from ml.config.types_hands import ALL_HANDS, HAND_TO_ID


def hand_range_to_vec169(rng: Dict[str, float]) -> np.ndarray:
    """Map {'AA':p, 'AKs':p, ...} to length-169 vector in ALL_HANDS order (normalized)."""
    y = np.zeros(len(ALL_HANDS), dtype=np.float32)
    s = 0.0
    for code, p in rng.items():
        if code in HAND_TO_ID:
            y[HAND_TO_ID[code]] += float(p)
            s += float(p)
    if s > 0:
        y /= s
    else:
        y[:] = 1.0 / len(ALL_HANDS)  # neutral fallback
    return y


def avg_vecs(vecs: List[np.ndarray]) -> np.ndarray:
    if not vecs:
        return np.ones(len(ALL_HANDS), dtype=np.float32) / len(ALL_HANDS)
    out = np.stack(vecs, axis=0).mean(axis=0)
    # guard tiny drift
    s = float(out.sum())
    if s > 0:
        out /= s
    return out.astype(np.float32)


def infer_node_key(row: pd.Series) -> str:
    """
    Produce a stable 'node_key' describing the post-flop state to learn.
    Priority: manifest.node_key → manifest.action_seq → street tag only.
    """
    if "node_key" in row and pd.notna(row["node_key"]):
        return str(row["node_key"])
    if "action_seq" in row and pd.notna(row["action_seq"]):
        return f"SEQ::{row['action_seq']}"
    # last-resort: street tag
    street = int(row.get("street", 1))
    return {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(street, f"STREET_{street}")
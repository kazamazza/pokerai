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

def to_vec169(rng) -> np.ndarray:
    """
    Convert a variety of range representations into a normalized 169-length vector.
    Accepts:
      - dict: {"AA":1.0, "AKs":0.5, ...}
      - list/ndarray length 169
      - 13x13 matrix (2D array)
      - JSON string of the above
    Returns:
      np.ndarray of shape (169,), dtype=float32, normalized so sum=1.
    """
    import numpy as np
    import json

    if isinstance(rng, str):
        try:
            rng = json.loads(rng)
        except Exception:
            return np.ones(169, dtype="float32") / 169.0

    if isinstance(rng, (list, tuple, np.ndarray)):
        arr = np.asarray(rng, dtype="float32")
        if arr.ndim == 2 and arr.shape == (13, 13):
            v = arr.reshape(169).astype("float32")
        elif arr.ndim == 1 and arr.size == 169:
            v = arr.astype("float32")
        else:
            # fallback: uniform
            v = np.ones(169, dtype="float32") / 169.0
    elif isinstance(rng, dict):
        v = np.zeros(169, dtype="float32")
        for h, w in rng.items():
            idx = HAND_TO_ID.get(str(h).upper())
            if idx is not None:
                try:
                    v[idx] = float(w)
                except Exception:
                    pass
    else:
        v = np.ones(169, dtype="float32") / 169.0

    # safety: clip negatives, renormalize
    v = np.clip(v, 0.0, None)
    s = float(v.sum())
    return (v / s) if s > 0 else (np.ones(169, dtype="float32") / 169.0)

from typing import Dict, List

# --- Canonical 169 hand list ---
# Order: pairs (AA..22), suited (AKs..32s), offsuit (AKo..32o)

ALL_HANDS: List[str] = []

# Pairs
for r in "AKQJT98765432":
    ALL_HANDS.append(r + r)

# Suited (rank1 > rank2)
for i, r1 in enumerate("AKQJT98765432"):
    for r2 in "AKQJT98765432"[i + 1:]:
        ALL_HANDS.append(r1 + r2 + "s")

# Offsuit (rank1 > rank2)
for i, r1 in enumerate("AKQJT98765432"):
    for r2 in "AKQJT98765432"[i + 1:]:
        ALL_HANDS.append(r1 + r2 + "o")

assert len(ALL_HANDS) == 169, f"Expected 169 hands, got {len(ALL_HANDS)}"

# --- Map hand string -> ID ---
HAND_TO_ID: Dict[str, int] = {h: i for i, h in enumerate(ALL_HANDS)}
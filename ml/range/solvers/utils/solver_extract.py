# ml/rangenet/io/solver_extract.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

from ml.range.solvers.utils.solver_json_extract import hand_to_index_169

def range_to_vec169(rmap: Dict[str, float]) -> np.ndarray:
    v = np.zeros(169, dtype=np.float32)
    for h, w in (rmap or {}).items():
        idx = hand_to_index_169(str(h))
        if idx is not None:
            try: v[idx] = float(w)
            except: pass
    return v

# ---------- JSON navigation ----------
def pick_root(doc: dict, node_key: str = "root") -> dict:
    if isinstance(doc.get("nodes"), dict) and isinstance(doc["nodes"].get(node_key), dict):
        return doc["nodes"][node_key]
    for k in ("root", "tree"):
        if isinstance(doc.get(k), dict):
            return doc[k]
    return doc

# ---------- policy extraction ----------
def _sum_over_children(strat_map: dict, child_labels: list[str], prefix: str) -> Optional[np.ndarray]:
    if not strat_map or not child_labels:
        return None
    want = str(prefix).upper()
    idxs = [i for i, lab in enumerate(child_labels) if str(lab).upper().startswith(want)]
    if not idxs:
        return None
    v = np.zeros(169, dtype=np.float32)
    for hand, probs in strat_map.items():
        hi = hand_to_index_169(str(hand))
        if hi is None:
            continue
        try:
            s = sum(float(probs[i]) for i in idxs)
            v[hi] = max(0.0, min(1.0, s))
        except Exception:
            pass
    return v if np.any(v) else None

def extract_action_vector_169(doc: dict, *, node_key: str, action_prefix: str) -> Optional[np.ndarray]:
    root = pick_root(doc, node_key=node_key)
    childrens = root.get("childrens") or {}
    child_labels = list(childrens.keys())
    strat_map = (root.get("strategy") or {}).get("strategy") or {}
    return _sum_over_children(strat_map, child_labels, action_prefix)

def extract_best_policy_169(doc: dict, *, node_key: str = "root") -> Tuple[Optional[np.ndarray], str]:
    """
    Try action prefixes in order → return first non-empty vector and a source tag.
    Order: BET → DONK → CHECK → fallback to 'RANGE' if root ranges exist.
    Returns: (vec, src) where src ∈ {"BET","DONK","CHECK","RANGE","NONE"}
    """
    root = pick_root(doc, node_key=node_key)
    childrens = root.get("childrens") or {}
    child_labels = list(childrens.keys())
    strat_map = (root.get("strategy") or {}).get("strategy") or {}

    for tag in ("BET", "DONK", "CHECK"):
        v = _sum_over_children(strat_map, child_labels, tag)
        if v is not None:
            return v, tag

    # Fallback: raw range payload shapes if present
    rmap = {}
    if isinstance(root.get("ranges"), dict):
        # prefer any side that exists; the consumer should know actor separately
        rmap = (root["ranges"].get("ip") or root["ranges"].get("oop") or {})
    elif isinstance(root.get("actors"), dict):
        act = root["actors"].get("ip") or root["actors"].get("oop") or {}
        rmap = act.get("range") or act.get("ranges") or {}
    if rmap:
        return range_to_vec169(rmap), "RANGE"

    return None, "NONE"
# ml/rangenet/sph_utils.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import re

# ----- rank/grid helpers -----
RANKS = "AKQJT98765432"
RANK_TO_I = {r: i for i, r in enumerate(RANKS)}

def _canon_card(card: str) -> Tuple[str, str]:
    """Return (rank, suit), ranks uppercased (e.g., 'Ah' -> ('A','h'))."""
    c = card.strip()
    if len(c) != 2:
        # Some SPH exports use '2h2d' (two cards concatenated); caller splits those.
        raise ValueError(f"Bad card token: {card}")
    return c[0].upper(), c[1].lower()

def _hand_key_from_combo(combo: str) -> str:
    """
    Combo like 'AhKd' / 'KdAh' / '2h2d' → canonical 2-card hand key:
      'AKs', 'AKo', '22'
    """
    c = combo.strip()
    # Normalize ordering: it may already be two cards back-to-back (e.g. 'AhKd')
    if len(c) != 4:
        raise ValueError(f"Bad combo token (expect 4 chars): {combo}")
    r1, s1 = _canon_card(c[:2])
    r2, s2 = _canon_card(c[2:])

    # Order by rank strength (A>K>...>2): higher rank first
    i1, i2 = RANK_TO_I[r1], RANK_TO_I[r2]
    if i1 < i2:
        hi_r, hi_s, lo_r, lo_s = r1, s1, r2, s2
    else:
        hi_r, hi_s, lo_r, lo_s = r2, s2, r1, s1

    if hi_r == lo_r:     # pair
        return f"{hi_r}{lo_r}"
    suited = (hi_s == lo_s)
    return f"{hi_r}{lo_r}{'s' if suited else 'o'}"

def _grid_index_from_key(key: str) -> int:
    """
    Map canonical key → index 0..168 in row-major 13×13 grid:
      rows/cols ordered by RANKS (A..2),
      row i == first rank, col j == second rank,
      above diagonal (i<j) are suited, below diagonal (i>j) are offsuit.
    """
    key = key.strip().upper()
    if len(key) == 2:  # pair, e.g. 'TT'
        r = key[0]; c = key[1]
        i = RANK_TO_I[r]; j = RANK_TO_I[c]
        return i * 13 + j
    if len(key) == 3:  # e.g. 'AKS' or 'AKO'
        r = key[0]; c = key[1]; t = key[2]
        i = RANK_TO_I[r]; j = RANK_TO_I[c]
        # Our convention: i==row (first rank), j==col (second rank)
        # For suited we expect i<j in the chart’s upper triangle;
        # for offsuit i>j in the lower triangle. If not, swap.
        if t == 'S' and not (i < j):
            i, j = min(i, j), max(i, j)
        if t == 'O' and not (i > j):
            i, j = max(i, j), min(i, j)
        return i * 13 + j
    raise ValueError(f"Bad hand key: {key}")

def zeros_169() -> np.ndarray:
    """Return 169-length float32 zeros (flat 13×13 grid)."""
    return np.zeros(169, dtype=np.float32)

# ----- SPH node → grid -----

def _try_node_range_array(node: dict) -> Optional[np.ndarray]:
    """
    If node already carries a compact 169-range (rare), accept it.
    Supports node['range'] or node['Range'] as list-like.
    """
    for k in ("range", "Range"):
        if k in node and isinstance(node[k], (list, tuple)) and len(node[k]) == 169:
            arr = np.asarray(node[k], dtype=np.float32)
            # if values look like percents (>1), scale down
            if arr.max(initial=0.0) > 1.001:
                arr = arr / 100.0
            return arr
    return None

def _parse_abs_percent(v) -> Optional[float]:
    """
    Accepts:
      - "Abs":"100.00" (string percent)
      - "Abs.%":"37.42" (string)
      - dicts with keys 'Abs' or 'Abs.%'
    Returns 0..1 or None.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x/100.0 if x > 1.001 else x
    s = str(v).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    try:
        x = float(s)
        return x/100.0 if x > 1.001 else x
    except Exception:
        return None

def node_hands_to_grid(node: dict) -> np.ndarray:
    """
    Convert an SPH 'Copy Node Strategy' JSON node into a 169-length array (float32, 0..1),
    averaging per-combo 'Abs.%' within each starting-hand cell.

    Expected shape examples:
      node = {
        "Name": "BB Call 2.50",
        "Hands": [
           {"Cards":"AhKd","Abs":"37.50","Played":"100.00","EV":"..."},
           {"Cards":"AdKh","Abs":"41.00", ...},
           ...
        ],
        "Nodes": [... children ...]
      }
    """
    # 1) fast path if node already has a 169-range
    arr = _try_node_range_array(node)
    if arr is not None:
        return arr.astype(np.float32, copy=False)

    hands = node.get("Hands") or node.get("hands") or []
    if not hands:
        # No hands list: treat as zero grid rather than crash
        return zeros_169()

    sums = np.zeros(169, dtype=np.float64)
    cnts = np.zeros(169, dtype=np.int32)

    for h in hands:
        # Cards may be '2h2d' or with spacing; normalize
        cards = str(h.get("Cards") or h.get("cards") or "").strip()
        cards = cards.replace(" ", "")
        if not cards or len(cards) != 4:
            continue
        key = _hand_key_from_combo(cards)

        # Pull Abs (absolute frequency for THIS action), 0..1
        # Try common keys
        abs_val = None
        if "Abs" in h:
            abs_val = _parse_abs_percent(h["Abs"])
        if abs_val is None and "Abs.%" in h:
            abs_val = _parse_abs_percent(h["Abs.%"])
        if abs_val is None:
            # Some exports store under 'Strategy' or 'Freq'
            for k in ("Strategy", "Freq", "Frequency"):
                if k in h:
                    abs_val = _parse_abs_percent(h[k]); break
        if abs_val is None:
            continue

        idx = _grid_index_from_key(key)
        sums[idx] += float(abs_val)
        cnts[idx] += 1

    out = np.zeros(169, dtype=np.float32)
    mask = cnts > 0
    out[mask] = (sums[mask] / cnts[mask]).astype(np.float32)

    # clamp just in case
    np.clip(out, 0.0, 1.0, out=out)
    return out
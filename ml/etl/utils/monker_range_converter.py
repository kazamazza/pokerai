import json
import re
import numpy as np

from ml.range.solvers.utils.range_utils import vec169_to_monker_string, hand_to_index


def _looks_like_monker_string(s: str) -> bool:
    # Heuristic: has hand tags and colons (e.g., "AA:0.5,AKs:0.25,...")
    if ":" not in s:
        return False
    return any(tag in s for tag in ("AA", "AKs", "KQo", "72o"))

def _parse_flat_numbers(s: str):
    """
    Try to parse a flat sequence of 169 numbers from a string (comma/space separated).
    Supports percent tokens like '12.5%'.
    Returns list[float] len=169 or None.
    """
    toks = re.split(r"[,\s]+", s.strip())
    nums = []
    for t in toks:
        if not t:
            continue
        if t.endswith("%"):
            try:
                nums.append(float(t[:-1]) / 100.0)
            except Exception:
                return None
        else:
            try:
                nums.append(float(t))
            except Exception:
                return None
    if len(nums) == 169:
        return nums
    return None

def _arr_to_monker(arr_like) -> str:
    """
    Accepts list/np.array of shape (169,) or (13,13) and emits Monker CSV.
    """
    a = np.asarray(arr_like, dtype=float)
    if a.ndim == 2 and a.shape == (13, 13):
        a = a.reshape(169)
    elif a.ndim == 1 and a.size == 169:
        pass
    else:
        raise ValueError(f"_arr_to_monker expected 169 or 13x13, got {a.shape}")
    a = np.clip(a, 0.0, 1.0)
    return vec169_to_monker_string(a)  # <-- your existing util

# --- main normalizer ---

def to_monker(range_payload) -> str:
    """
    Normalize a range payload to Monker string:
      - Monker-like string → return as-is
      - 169/13x13 arrays/lists → convert
      - JSON list/dict (ip/oop/range/grid/matrix/weights/data) → convert
      - Flat '169 numbers' string (supports %) → convert
      - Else: return original string
    """
    # Fast path: array-like
    if isinstance(range_payload, (list, tuple, np.ndarray)):
        return _arr_to_monker(range_payload)

    # Dict container
    if isinstance(range_payload, dict):
        for k in ("ip", "oop", "range", "grid", "matrix", "weights", "data"):
            if k in range_payload:
                return _arr_to_monker(range_payload[k])
        return str(range_payload)

    # String cases
    if isinstance(range_payload, str):
        s = range_payload.strip()

        if _looks_like_monker_string(s):
            return s

        # JSON?
        if s.startswith("{") or s.startswith("["):
            try:
                obj = json.loads(s)
                return to_monker(obj)  # recurse
            except Exception:
                # fall through
                pass

        # Flat 169 numbers?
        nums = _parse_flat_numbers(s)
        if nums is not None:
            return _arr_to_monker(nums)

        # Fallback: keep as-is
        return s

    # Anything else → stringify
    return str(range_payload)

def monker_to_vec169(monker_str: str) -> np.ndarray:
    """Convert Monker AA:0.5,AKs:0.25,... string to a 169 vector."""

    vals = np.zeros(169, dtype=np.float32)
    for tok in re.split(r"[,\s]+", monker_str.strip()):
        if not tok or ":" not in tok:
            continue
        hand, v = tok.split(":", 1)
        vals[hand_to_index(hand.strip())] = float(v)
    return vals
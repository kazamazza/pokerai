import numpy as np
import re, json
from pathlib import Path

from ml.config.types_hands import RANKS


def zeros_169():
    return np.zeros(169, dtype=np.float32)

def parse_range_text_to_grid(path: Path) -> np.ndarray:
    """
    Parse various SPH/Monker export formats into a flat 169-vector.
    Supports:
      - 13x13 CSV or whitespace grid
      - Flat list of 169 numbers
      - CARD:VALUE dict-style
      - [xx.xx]HandGroup[/xx.xx] syntax
    Returns: np.ndarray of shape (169,), dtype=float32
    """
    txt = Path(path).read_text(encoding="utf-8").strip()

    # --- Try JSON first ---
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and len(obj) == 169:
            return np.array(obj, dtype=np.float32)
        if isinstance(obj, dict):
            # dict may be {"AA":1.0,...} or {"range":[...]}
            if "range" in obj and len(obj["range"]) == 169:
                return np.array(obj["range"], dtype=np.float32)
            # CARD:VALUE dict
            vals = [0.0] * 169
            for k, v in obj.items():
                idx = _hand_to_index(k)
                vals[idx] = float(v)
            if sum(vals) > 0:
                return np.array(vals, dtype=np.float32)
    except Exception:
        pass

    # --- CARD:VALUE text (AA:1.0,A2s:0.024,...) ---
    if ":" in txt and any(h in txt for h in ["AA", "AKs", "72o"]):
        vals = [0.0] * 169
        for tok in re.split(r"[,\s]+", txt):
            if not tok or ":" not in tok:
                continue
            hand, val = tok.split(":")
            idx = _hand_to_index(hand)
            vals[idx] = float(val)
        return np.array(vals, dtype=np.float32)

    # --- SPH [xx.xx]cards[/xx.xx] groups ---
    if "[" in txt and "]" in txt and "/" in txt:
        vals = [0.0] * 169
        for group in re.finditer(r"\[(.*?)\](.*?)\[/\1\]", txt):
            val = float(group.group(1))
            hands = re.split(r"[, ]+", group.group(2).strip())
            for h in hands:
                if h:
                    idx = _hand_to_index_compact(h)
                    vals[idx] = val / 100.0 if val > 1.0 else val
        return np.array(vals, dtype=np.float32)

    # --- Flat list of numbers ---
    toks = re.split(r"[,\s]+", txt)
    nums = []
    for t in toks:
        if not t:
            continue
        if t.endswith("%"):
            nums.append(float(t[:-1]) / 100.0)
        else:
            nums.append(float(t))
    if len(nums) == 169:
        return np.array(nums, dtype=np.float32)

    raise ValueError(f"Unrecognized range format in {path} (len={len(txt)})")

# --- helper: map "AKo"/"AKs"/"AA" → 0..168 index ---
def _hand_to_index(hand: str) -> int:
    hand = hand.strip()
    if len(hand) == 2:  # pair like "AA"
        r = RANKS.index(hand[0])
        return r * 13 + r
    if len(hand) == 3:
        r1, r2, s = hand[0], hand[1], hand[2]
        i = RANKS.index(r1)
        j = RANKS.index(r2)
        if s == "s":
            return i * 13 + j
        elif s == "o":
            return j * 13 + i
    raise ValueError(f"Bad hand code: {hand}")

# --- helper: map expanded cards ("AhKh") → compact index ---
def _hand_to_index_compact(cards: str) -> int:
    """
    Collapse e.g. AhKh → AKs, AdKc → AKo, etc.
    """
    if len(cards) != 4:
        raise ValueError(f"Bad card spec: {cards}")
    r1, s1, r2, s2 = cards[0], cards[1], cards[2], cards[3]
    if r1 == r2:
        return _hand_to_index(r1 + r2)
    suited = (s1 == s2)
    high, low = (r1, r2) if RANKS.index(r1) < RANKS.index(r2) else (r2, r1)
    code = high + low + ("s" if suited else "o")
    return _hand_to_index(code)
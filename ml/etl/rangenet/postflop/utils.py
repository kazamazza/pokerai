import gzip
import json
import math
import re
from typing import Dict, Any, List, Sequence, Optional, Tuple

from ml.models.policy_consts import VOCAB_INDEX

Json = Dict[str, Any]

BET_PCT_RE   = re.compile(r"\bbet\b\s*([0-9]+(?:\.[0-9]+)?)\s*%+", re.IGNORECASE)
BET_NUM_RE   = re.compile(r"\bbet\b\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)  # absolute bb (no %)
RAISE_NUM_RE = re.compile(r"\braise(?:\s*to)?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

_NUMERIC_KEYS_POT  = ["pot_bb","total_pot_bb","pot","total_pot","round_pot","potSize","pot_size","current_pot"]
_NUMERIC_KEYS_FACE = ["to_call","call_amount","facing_bet_bb","facing_bet","amount_to_call","last_bet","bet_to_call"]

def load_solver(path: str) -> Json:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def root_node(payload: Json) -> Json:
    for key in ("root","node","game_tree","tree","payload"):
        v = payload.get(key)
        if isinstance(v, dict): return v
    return payload

# ---- Tree utils ----
def _child_container(node: Json):
    for key in ("children","childrens","nodes"):
        v = node.get(key)
        if isinstance(v, list): return "list", v
        if isinstance(v, dict): return "dict", v
    return None, None

def get_children(node: Json) -> Dict[str, Json]:
    kind, cont = _child_container(node)
    if kind == "list":
        out: Dict[str, Json] = {}
        for ch in cont:  # type: ignore
            if isinstance(ch, dict):
                lab = ch.get("action") or ch.get("label") or ch.get("name")
                if isinstance(lab, str): out[lab] = ch
        return out
    if kind == "dict":
        return {str(k):v for k,v in cont.items() if isinstance(v, dict)}  # type: ignore
    return {}

def action_list(node: Json) -> List[str]:
    for key in ("actions","available_actions","menu"):
        v = node.get(key)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v  # type: ignore
    return list(get_children(node).keys())


def _pick_number(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if math.isfinite(v): return v
            except Exception:
                pass
    return None

def _extract_state_from_node(node: Json) -> Tuple[Optional[float], Optional[float]]:
    objs = [node]
    for k in ("state","info","stats","meta","ctx","context"):
        v = node.get(k)
        if isinstance(v, dict): objs.append(v)
    pot = face = None
    for obj in objs:
        pot  = pot  or _pick_number(obj, _NUMERIC_KEYS_POT)
        face = face or _pick_number(obj, _NUMERIC_KEYS_FACE)
    return pot, face

# ---- Label parsing ----
def parse_bet_pct(label: str) -> Optional[float]:
    m = BET_PCT_RE.search(label);
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

def parse_bet_amount_bb(label: str) -> Optional[float]:
    if "%" in label: return None
    m = BET_NUM_RE.search(label)
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

def parse_raise_to_bb(label: str) -> Optional[float]:
    m = RAISE_NUM_RE.search(label)
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

# ---- Bucketing ----
def _bucket_bet_pct(pct: Optional[float]) -> Optional[str]:
    """Map a % of pot to a bet token with tolerant ranges."""
    if pct is None:
        return None

    # Convert to fraction (0–inf), clamp a bit to avoid crazy values from rounding
    x = max(0.0, float(pct) / 100.0)

    # Tolerant ranges around common sizes
    # (low, high, label)
    RANGES = [
        (0.20, 0.30, "BET_25"),   # ~25%
        (0.30, 0.36, "BET_33"),   # ~33%
        (0.47, 0.53, "BET_50"),   # ~50%
        (0.63, 0.69, "BET_66"),   # ~66%
        (0.73, 0.77, "BET_75"),   # ~75%
        (0.95, 1.05, "BET_100"),  # ~100%
    ]

    for lo, hi, label in RANGES:
        if lo <= x <= hi:
            return label

    # Fallback: nearest center if within a soft tolerance (helps on 0.499, 0.505, 1.08, etc.)
    CENTERS = [
        (0.25, "BET_25"),
        (0.33, "BET_33"),
        (0.50, "BET_50"),
        (0.66, "BET_66"),
        (0.75, "BET_75"),
        (1.00, "BET_100"),
    ]
    nearest = min(CENTERS, key=lambda c: abs(x - c[0]))
    if abs(x - nearest[0]) <= 0.06:  # 6% pot tolerance
        return nearest[1]

    return None

_TARGETS = [1.5, 2.0, 3.0, 4.0, 5.0]

def _nearest_raise_label(rb: float) -> str:
    t = min(_TARGETS, key=lambda x: abs(rb - x))
    if t == 1.5: return "RAISE_150"
    if t == 2.0: return "RAISE_200"
    if t == 3.0: return "RAISE_300"
    if t == 4.0: return "RAISE_400"
    return "RAISE_500"

def _infer_facing(menu_pcts: Sequence[float], pot_before: float, raise_to_bb: Optional[float]) -> float:
    if not menu_pcts or raise_to_bb is None or not math.isfinite(raise_to_bb):
        return 0.33 * pot_before
    best = None
    for p in menu_pcts:
        face = p * pot_before
        rb = raise_to_bb / max(face, 1e-6)
        score = min(abs(rb - t) for t in _TARGETS)
        if best is None or score < best[0]:
            best = (score, face)
    return best[1] if best else 0.33 * pot_before

def _bucket_raise(raise_to_bb: Optional[float], facing_bet_bb: float, pot_before: float, stack_bb: float) -> str:
    if raise_to_bb is None or not math.isfinite(raise_to_bb): return "RAISE_200"
    if raise_to_bb >= (stack_bb - 0.5): return "ALLIN"
    rb = raise_to_bb / max(facing_bet_bb, 1e-6)
    return _nearest_raise_label(rb)

def _bump(counts: List[float], tok: str, inc: float = 1.0) -> None:
    i = VOCAB_INDEX.get(tok)
    if i is not None: counts[i] += inc


def _bump_vec(probs: Dict[str, float], key: Optional[str], val: float) -> None:
    if key in probs:
        probs[key] += float(val)

def _first_label_containing(children: Dict[str, Any], token: str) -> Optional[str]:
    t = token.lower()
    for lab in children.keys():
        if t in str(lab).lower():
            return lab
    return None

def _parse_number_in_label(label: str) -> Optional[float]:
    import re
    m = re.search(r'([-+]?\d+(?:\.\d+)?)', str(label))
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

def _compute_facing_from_label(label: str, pot_bb: float) -> float:
    pct = parse_bet_pct(label)
    if pct is not None:
        return (pct / 100.0) * pot_bb
    amt = _parse_number_in_label(label)
    return float(amt) if amt is not None else 0.0
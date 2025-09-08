# ---- canonical positions (6-max) ----
import numpy as np

from ml.range.solvers.utils.range_utils import hand_to_index

POS_ORDER = ["UTG","HJ","CO","BTN","SB","BB"]
POS_SET   = set(POS_ORDER)
POS_IDX   = {p:i for i,p in enumerate(POS_ORDER)}

def canon_pos(p: str | None) -> str | None:
    if not p or not isinstance(p, str):
        return None
    p = p.strip().upper()
    alias = {"BU":"BTN","MP":"HJ","EP":"UTG","LJ":"HJ"}
    p = alias.get(p, p)
    return p if p in POS_SET else None


# ---------------- nearest stack ----------------
def nearest_stack(target: float | int, available: list[int]) -> int:
    """
    Pick the nearest available stack (tie → smaller).
    """
    t = float(target)
    return int(min(available, key=lambda s: (abs(s - t), s)))


# ---------------- opener substitution candidates ----------------
def _one_step_neighbors(ip: str) -> list[str]:
    """Immediate neighbors in open-order (only RFI seats)."""
    i = POS_IDX[ip]
    nbrs = []
    if i - 1 >= 0: nbrs.append(POS_ORDER[i-1])
    if i + 1 < len(POS_ORDER): nbrs.append(POS_ORDER[i+1])
    # Only allow opens from RFI seats, not blinds
    return [p for p in nbrs if p in ("UTG","HJ","CO","BTN")]

def _candidate_pairs(ip: str, oop: str, allow_pair_subs: bool) -> list[tuple[str,str,int,bool]]:
    """
    Return candidate (ip, oop, fallback_level, substituted?)
    level=0 → exact, level=2 → opener 1-step substitution (same OOP).
    """
    ip = canon_pos(ip); oop = canon_pos(oop)
    out: list[tuple[str,str,int,bool]] = []
    if not ip or not oop: return out
    out.append((ip, oop, 0, False))
    if allow_pair_subs and ip in ("UTG","HJ","CO","BTN"):
        for ip2 in _one_step_neighbors(ip):
            out.append((ip2, oop, 2, True))
    return out


# ---------------- SRP detection (Monker raw tokens) ----------------
import re
PERCENT_RE = re.compile(r"^\d+%$")

# raw vendor tokens we’ve seen
OPEN_ACTIONS    = {"Min", "AI"}          # open / min-raise / all-in treated as opening raise
RAISEY_ACTIONS  = {"Min", "AI", "3sb"}   # anything that re-raises preflop (keep vendor token)
CALL_ACTION     = "Call"
FOLD_ACTION     = "Fold"

ACTION_NORMALIZE = {
    "Min": "RAISE",
    "AI": "ALL_IN",
    "3sb": "3BET",
    "Call": "CALL",
    "Fold": "FOLD",
    "Open": "OPEN",
    "Limp": "LIMP",
    "Bet": "BET",
    "Raise": "RAISE",
    "Check": "CHECK",
    "Cbet": "CBET",
    "Donk": "DONK",
    "3Bet": "3BET",
    "4Bet": "4BET",
    "5Bet": "5BET",
}

def _is_fold_token(act: str | None) -> bool:
    return act in {"Fold","FOLD"}

def _is_raise_token(act: str | None) -> bool:
    if not act: return False
    if act in {"Min","AI","3sb"} or PERCENT_RE.match(act):
        return True
    return act.upper() in {"RAISE","ALL_IN","OPEN","LIMP","3BET","4BET","5BET"}

def first_non_fold_opener(seq_raw: list[dict]) -> tuple[str|None, str|None]:
    for e in seq_raw:
        pos = canon_pos(e.get("pos"))
        act = e.get("action")
        if not pos:
            continue
        if _is_fold_token(act):
            continue
        if _is_raise_token(act):
            return pos, act
    return None, None

def defender_first_action(seq_raw: list[dict], defender_pos: str) -> str | None:
    for e in seq_raw:
        if canon_pos(e.get("pos")) == defender_pos and "action" in e:
            return e["action"]
    return None

def is_srp_open_call(seq_raw: list[dict], ip_pos: str, oop_pos: str) -> bool:
    opener_pos, opener_act = first_non_fold_opener(seq_raw)
    if opener_pos != ip_pos:
        return False
    if opener_act not in OPEN_ACTIONS and not PERCENT_RE.match(opener_act or ""):
        return False

    re_raised = False
    for e in seq_raw[1:]:
        pos = canon_pos(e.get("pos"))
        act = e.get("action")
        if _is_raise_token(act):  # any raise before defender acts disqualifies SRP open/call
            re_raised = True
        if pos == oop_pos:
            return (act == CALL_ACTION) and (not re_raised)
    return False


# ---------------- vendor range loader ----------------
from pathlib import Path
import json

import json
import re
from pathlib import Path
from typing import List

def _canonical_169_keys() -> List[str]:
    """Return standard 169-grid keys in A..2 × A..2 order:
       rows = first rank A..2, cols = second rank A..2
       upper-triangle (i<j) suited 'AKs', diagonal pairs 'AA', lower (i>j) offsuit 'AKo'."""
    ranks = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
    keys: List[str] = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                keys.append(f"{r1}{r2}")        # pair
            elif i < j:
                keys.append(f"{r1}{r2}s")       # suited (upper triangle)
            else:
                keys.append(f"{r1}{r2}o")       # offsuit (lower triangle)
    return keys

def _load_vendor_range_compact(path: Path) -> str:
    """
    Load a vendor range file and return a JSON string with 169 floats in canonical order.
    Supported formats:
      - JSON list of 169 floats
      - JSON dict with key 'range' → 169 floats
      - CSV/whitespace list of 169 bare numbers
      - CSV of label:value pairs (e.g. 'AA:1.0,A2s:0.024,...') — labels must be 169 hand keys.
      - Same as above but percentages like 'AKs:12.5%' (converted to 0.125)
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip()

    # 1) Try JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "range" in obj and isinstance(obj["range"], list) and len(obj["range"]) == 169:
            return json.dumps([float(x) for x in obj["range"]])
        if isinstance(obj, list) and len(obj) == 169:
            return json.dumps([float(x) for x in obj])
    except Exception:
        pass

    # 2) Try labeled CSV/dict: "AA:1.0,A2s:0.024,..." (possibly with whitespace/newlines)
    #    We accept lines/tokens separated by commas or whitespace.
    #    Each token can be "KEY:VALUE" or just "VALUE".
    tokens = re.split(r"[,\s]+", txt)
    if any(":" in t for t in tokens if t):
        kv = {}
        for t in tokens:
            if not t or ":" not in t:
                continue
            k, v = t.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            try:
                if v.endswith("%"):
                    val = float(v[:-1]) / 100.0
                else:
                    val = float(v)
                kv[k] = val
            except Exception:
                # ignore malformed entries
                continue
        # If we collected labeled values, map them into canonical order
        if kv:
            keys = _canonical_169_keys()
            arr = [float(kv.get(k, 0.0)) for k in keys]
            # If we want to be strict, ensure coverage:
            # if len([k for k in keys if k in kv]) != 169: raise ValueError(...)
            return json.dumps(arr)

    # 3) Try plain numeric CSV/whitespace (169 numbers)
    nums = []
    for t in tokens:
        if not t:
            continue
        try:
            if t.endswith("%"):
                nums.append(float(t[:-1]) / 100.0)
            else:
                nums.append(float(t))
        except Exception:
            # skip non-numeric tokens (e.g., keys from an unexpected format)
            continue
    if len(nums) == 169:
        return json.dumps(nums)

    raise ValueError(f"Unrecognized vendor range format or wrong length at {path}")

def monker_string_to_vec169(s: str) -> np.ndarray:
    """
    Parse Monker string like:
      'AA:1.0,AKs:0.75,AKo:0.2,...'
    Returns float32 vector of shape (169,) clamped to [0,1].
    Ignores tokens without ':' and silently skips malformed hands.
    """
    vals = np.zeros(169, dtype=np.float32)
    if not s:
        return vals
    # split on commas and whitespace
    tokens = re.split(r"[,\s]+", s.strip())
    for tok in tokens:
        if not tok or ":" not in tok:
            continue
        hand, v = tok.split(":", 1)
        hand = hand.strip()
        v = v.strip().rstrip("%")
        try:
            idx = hand_to_index(hand)
            val = float(v)
            if val > 1.0:  # allow percents given as 0..100
                val = val / 100.0
            vals[idx] = np.clip(val, 0.0, 1.0)
        except Exception:
            # skip unrecognized tokens silently
            continue
    return vals
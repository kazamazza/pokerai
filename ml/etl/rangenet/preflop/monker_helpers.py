# ml/etl/rangenet/preflop/monker_helpers.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# Canonical positions used by vendor in your packs
POS_SET = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}

# Raw vendor action tokens (from probe); we keep them raw for matching filenames
OPEN_ACTIONS = {"Min", "AI"}        # vendor "open/raise" family
RAISEY_ACTIONS = {"Min", "AI", "3sb"} # anything that re-raises before defender acts
CALL_ACTION = "Call"
FOLD_ACTION = "Fold"

ACTION_NORMALIZE = {
    "Min": "RAISE",
    "AI": "ALL_IN",
    "3sb": "3BET",     # keep it if you see it
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

def canon_action(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return ACTION_NORMALIZE.get(raw, raw.upper())

def canon_pos(p: str) -> Optional[str]:
    """Normalize/validate position token to vendor canon (returns None if unknown)."""
    if not isinstance(p, str):
        return None
    p = p.strip().upper()
    # map aliases here if you ever see them (e.g., BU->BTN, MP->HJ/LJ, etc.)
    alias = {"BU": "BTN", "MP": "HJ", "EP": "UTG", "LJ": "HJ"}  # adjust if needed
    p = alias.get(p, p)
    return p if p in POS_SET else None

def parse_seq_from_stem(stem: str) -> List[Dict[str, str]]:
    """
    Parse filename stem into [{"pos": "UTG", "action": "Min"}, ...] using RAW vendor tokens.
    We do NOT normalize actions; we want to match the vendor taxonomy exactly.
    """
    toks = stem.split("_")
    seq: List[Dict[str, str]] = []
    i = 0
    while i < len(toks):
        pos = canon_pos(toks[i])
        if not pos:
            i += 1
            continue
        action = None
        if i + 1 < len(toks) and not canon_pos(toks[i + 1]):
            action = toks[i + 1]  # keep raw like "Min", "AI", "Call", "Fold", "3sb"
            i += 2
        else:
            i += 1
        e = {"pos": pos}
        if action is not None:
            e["action"] = action
        seq.append(e)
    return seq

def first_non_fold_opener(seq: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Return (pos, raw_action) of the first *opener* (Min/AI)."""
    for e in seq:
        act = e.get("action")
        if act in OPEN_ACTIONS:
            return e["pos"], act
    return None, None

def defender_first_action(seq: List[Dict[str, str]], defender_pos: str) -> Optional[str]:
    """Return the defender's first raw action token if present."""
    for e in seq:
        if e.get("pos") == defender_pos and "action" in e:
            return e["action"]
    return None

def is_srp_open_call(seq: List[Dict[str, str]], ip_pos: str, oop_pos: str) -> bool:
    """
    Detect Single-Raised Pot: opener = ip_pos with Min/AI; defender = oop_pos with first action Call;
    and no intermediate re-raise before defender acts.
    """
    if not seq:
        return False
    opener = first_non_fold_opener(seq)
    if opener == (None, None):
        return False

    open_pos, open_act = opener
    if open_pos != ip_pos or open_act not in OPEN_ACTIONS:
        return False

    re_raised = False
    for e in seq[1:]:
        pos = e.get("pos")
        act = e.get("action")
        if act in RAISEY_ACTIONS:
            re_raised = True
        if pos == oop_pos:
            return (act == CALL_ACTION) and (not re_raised)
    return False

def vendor_stem_for_pair(seq: List[Dict[str, str]], ip_pos: str, oop_pos: str) -> Optional[str]:
    """
    Build a compact, vendor-raw stem for a pair: e.g. 'BTN_Min_BB_Call' or 'CO_AI_BB_Call'.
    Returns None if pattern doesn't fit SRP open/call.
    """
    if not is_srp_open_call(seq, ip_pos, oop_pos):
        return None
    # opener (first non-fold)
    open_pos, open_act = first_non_fold_opener(seq)
    # defender first action
    oop_act = defender_first_action(seq, oop_pos)
    if open_pos and open_act and oop_act:
        return f"{open_pos}_{open_act}_{oop_pos}_{oop_act}"
    return None

def nearest_stack(target: float, available: List[int]) -> int:
    """Pick nearest stack (tie -> smaller)."""
    target = float(target)
    best = min(available, key=lambda s: (abs(s - target), s))
    return int(best)
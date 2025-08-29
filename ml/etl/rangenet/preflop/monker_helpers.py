from __future__ import annotations

import re
from typing import Optional, List, Dict, Tuple, Any

PERCENT_RE = re.compile(r"^\d+%$")

POS_SET = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}

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

def canon_action(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    if PERCENT_RE.match(raw):     # e.g. "60%", "125%"
        return "RAISE"
    return ACTION_NORMALIZE.get(raw, raw.upper())

def canon_pos(p: str) -> Optional[str]:
    if not isinstance(p, str):
        return None
    p = p.strip().upper()
    alias = {"BU": "BTN", "MP": "HJ", "EP": "UTG", "LJ": "HJ"}
    p = alias.get(p, p)
    return p if p in POS_SET else None

def parse_seq_from_stem(stem: str) -> List[Dict[str, str]]:
    """
    Parse vendor stem → list of dicts with RAW vendor tokens:
      [{"pos":"UTG","action":"60%"}, {"pos":"HJ","action":"Fold"}, ...]
    Positions are canonicalized; actions are left raw (Min/AI/3sb/60%/Call/Fold...).
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
            action = toks[i + 1]  # raw token (Min/AI/3sb/60%/Call/Fold/...)
            i += 2
        else:
            i += 1
        e = {"pos": pos}
        if action is not None:
            e["action"] = action
        seq.append(e)
    return seq

# ---- raw token predicates (recognize vendor % as raise) ----

def _is_raise_token_raw(a: Optional[str]) -> bool:
    if not a:
        return False
    return a in {"Min", "AI", "3sb"} or PERCENT_RE.match(a)

def _is_call_token_raw(a: Optional[str]) -> bool:
    return a in {"Call", "CALL"}

def _is_fold_token_raw(a: Optional[str]) -> bool:
    return a in {"Fold", "FOLD"}

# ---- simple open/defender helpers on RAW sequence ----

def first_non_fold_opener(seq_raw: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    First actor whose action is not a pure Fold AND is raise/open-ish (includes vendor % raise).
    Returns raw action (e.g., 'Min','AI','60%').
    """
    for e in seq_raw:
        a = e.get("action")
        if a and not _is_fold_token_raw(a) and _is_raise_token_raw(a):
            return e["pos"], a
    return None, None

def defender_first_action_raw(seq_raw: List[Dict[str, str]], defender_pos: str) -> Optional[str]:
    for e in seq_raw:
        if e.get("pos") == defender_pos and "action" in e:
            return e["action"]
    return None

def unique_seen_positions(seq_raw: List[Dict[str, str]]) -> List[str]:
    seen = []
    for e in seq_raw:
        p = e.get("pos")
        if p and p not in seen:
            seen.append(p)
    return seen

# ---- context classifier (coarse but robust & vendor-agnostic) ----

def classify_context(seq_raw: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Compute coarse preflop context:
      - ctx: one of {'LIMPED_SINGLE','LIMPED_MULTI','VS_OPEN','VS_3BET','VS_4BET'} (None if unknown)
      - raise_depth: number of raise-ish events in the whole sequence
      - limp_count: number of 'Limp' before the first raise-ish
      - multiway: True if >= 2 distinct callers after first raise-ish (before any re-raise)
      - opener_pos_raw/action_raw (if any)
    This is enough to drive lookup grouping without filtering anything out.
    """
    # count limps until first raise-ish
    limp_count = 0
    opener_pos_raw, opener_action_raw = None, None
    for e in seq_raw:
        a = e.get("action")
        if _is_raise_token_raw(a):
            opener_pos_raw, opener_action_raw = e.get("pos"), a
            break
        if a == "Limp":
            limp_count += 1

    # how many raise-ish overall?
    raise_depth = sum(1 for e in seq_raw if _is_raise_token_raw(e.get("action")))

    # multiway heuristic: how many distinct CALLers after first raise-ish until next raise-ish?
    call_positions = set()
    if opener_pos_raw:
        after_open = False
        reraised = False
        for e in seq_raw:
            if not after_open:
                if e.get("pos") == opener_pos_raw and _is_raise_token_raw(e.get("action")):
                    after_open = True
                continue
            a = e.get("action")
            if _is_raise_token_raw(a):
                reraised = True
                break
            if _is_call_token_raw(a):
                p = e.get("pos")
                if p:
                    call_positions.add(p)
        multiway = len(call_positions) >= 2 and not reraised
    else:
        multiway = limp_count >= 2 and raise_depth == 0

    # coarse ctx
    if raise_depth == 0:
        ctx = "LIMPED_SINGLE" if limp_count == 1 else ("LIMPED_MULTI" if limp_count >= 2 else None)
    elif raise_depth == 1:
        ctx = "VS_OPEN"
    elif raise_depth == 2:
        ctx = "VS_3BET"
    else:
        ctx = "VS_4BET"

    return {
        "ctx": ctx,
        "raise_depth": int(raise_depth),
        "limp_count": int(limp_count),
        "multiway": bool(multiway),
        "opener_pos_raw": opener_pos_raw,
        "opener_action_raw": opener_action_raw,
    }
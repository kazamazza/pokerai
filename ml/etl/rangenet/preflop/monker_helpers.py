from __future__ import annotations
import re
from typing import Optional, List, Dict, Tuple, Any

from ml.etl.utils.range_lookup import POS_ORDER

SIZE_X_RE   = re.compile(r"^\d+(\.\d+)?x$", re.IGNORECASE)   # e.g., 2.5x, 9x
PERCENT_RE = re.compile(r"^\d+%$")

RAISE_CANON = {"OPEN","RAISE","BET","3BET","4BET","5BET","ALL_IN"}
RAISE_RAW   = {"Min","AI","3sb","Open","Raise","Bet","3Bet","4Bet","5Bet"}  # vendor tokens you’ve seen


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



def _is_raise_token_raw(a: Optional[str]) -> bool:
    if not a:
        return False
    # Fast-path for known vendor tokens (case-sensitive as provided)
    if a in RAISE_RAW:
        return True
    # Case-insensitive check on canonicalized action
    ac = canon_action(a)  # uses your ACTION_NORMALIZE (+ upper)
    if ac in RAISE_CANON:
        return True
    # Size-bearing tokens (2.5x, 60%) should count as raises too
    if SIZE_X_RE.match(a) or PERCENT_RE.match(a):
        return True
    return False

def _is_call_token_raw(a: Optional[str]) -> bool:
    return bool(a) and a.lower() == "call".lower()

def _is_fold_token_raw(a: Optional[str]) -> bool:
    return bool(a) and a.lower() == "fold".lower()

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
    Compute coarse preflop context from RAW vendor tokens.
    - Limp detection: treat 'Limp' OR 'Call' *before the first raise-ish* as a limp.
    - multiway: ≥2 distinct callers after open before any re-raise,
                or ≥2 distinct callers when no raise at all (pure limp pot).
    """
    # locate first raise-ish
    first_raise_idx = None
    opener_pos_raw = None
    opener_action_raw = None
    for i, e in enumerate(seq_raw):
        a = e.get("action")
        if _is_raise_token_raw(a):
            first_raise_idx = i
            opener_pos_raw = e.get("pos")
            opener_action_raw = a
            break

    # count limps = 'Limp' OR 'Call' BEFORE any raise
    limpers: List[str] = []
    scan_upto = first_raise_idx if first_raise_idx is not None else len(seq_raw)
    for e in seq_raw[:scan_upto]:
        a = e.get("action")
        p = e.get("pos")
        if a == "Limp" or _is_call_token_raw(a):
            if p:
                limpers.append(p)

    limp_count = len(set(limpers))

    # total raise depth
    raise_depth = sum(1 for e in seq_raw if _is_raise_token_raw(e.get("action")))

    # multiway heuristic
    if first_raise_idx is not None:
        # callers after open, before any re-raise
        call_positions = set()
        for e in seq_raw[first_raise_idx + 1:]:
            a = e.get("action")
            if _is_raise_token_raw(a):
                break
            if _is_call_token_raw(a):
                p = e.get("pos")
                if p:
                    call_positions.add(p)
        multiway = len(call_positions) >= 2
    else:
        # pure limp pot: multiway if ≥2 limpers
        multiway = limp_count >= 2

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

PAIR_PATTERNS = [
    re.compile(r"\b(UTG|HJ|CO|BTN|SB|BB)[\s_\-]*vs[\s_\-]*(UTG|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE),
    re.compile(r"\b(UTG|HJ|CO|BTN|SB|BB)[\s_\-]*(?:v|x)[\s_\-]*(UTG|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE),
    re.compile(r"\b(UTG|HJ|CO|BTN|SB|BB)[\s_\-]+(UTG|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE),
]

def find_hu_pair_in_text(text: str):
    # try explicit patterns like "BTN_vs_BB", "CO-BB"
    for pat in PAIR_PATTERNS:
        m = pat.search(text)
        if m:
            return canon_pos(m.group(1)), canon_pos(m.group(2))
    # fallback: first two distinct seat hits in order of appearance
    hits = []
    for p in POS_ORDER:
        for mm in re.finditer(fr"\b{p}\b", text, flags=re.IGNORECASE):
            hits.append((mm.start(), canon_pos(p)))
    hits.sort()
    uniq = []
    for _, p in hits:
        if p and p not in uniq:
            uniq.append(p)
        if len(uniq) == 2:
            break
    return (uniq[0], uniq[1]) if len(uniq) == 2 else (None, None)

def detect_raise_depth_from_text(text: str) -> int:
    t = text.lower()
    if "5bet" in t: return 3
    if "4bet" in t: return 3
    if "3bet" in t: return 2
    if "open" in t or "raise" in t or "bet" in t: return 1
    if "limp" in t: return 0
    return 0
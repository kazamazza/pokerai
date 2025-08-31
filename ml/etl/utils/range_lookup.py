# ---- canonical positions (6-max) ----
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
    """
    From a parsed filename token list like:
      [{"pos":"UTG","action":"Min"},{"pos":"HJ","action":"Fold"}, ...]
    return the first (pos, action) that is NOT Fold and is raise/open-ish.
    """
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
    """
    Single-raised pot pattern:
      - opener = ip_pos with a raise-like token (OPEN_ACTIONS or %)
      - before defender acts, no re-raise occurs
      - defender's first action is Call
    """
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

def _load_vendor_range_compact(path: Path) -> str:
    """
    Load a Monker vendor range into your compact internal representation.

    Contract:
      - Return a *string payload* (or bytes) your downstream expects
        (e.g., JSON text of a 169-length vector or compact encoding you already use).

    This generic loader tries:
      1) JSON file: either {"range":[...169 floats...]} or a raw list [...169...]
      2) Plain text/CSV: comma/whitespace separated 169 numbers
    Raise if shape is wrong so fallbacks can trigger.
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip()

    # Try JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "range" in obj and len(obj["range"]) == 169:
            return json.dumps(obj["range"])  # unify to raw list payload
        if isinstance(obj, list) and len(obj) == 169:
            return json.dumps(obj)
    except Exception:
        pass

    # Try CSV / whitespace
    # Grab all tokens that look like numbers; allow percents to be converted
    import re
    toks = re.split(r"[,\s]+", txt)
    nums = []
    for t in toks:
        if not t:
            continue
        if t.endswith("%"):
            try:
                nums.append(float(t[:-1]) / 100.0)
            except Exception:
                continue
        else:
            try:
                nums.append(float(t))
            except Exception:
                continue
    if len(nums) == 169:
        return json.dumps(nums)

    raise ValueError(f"Unrecognized vendor range format or wrong length at {path}")
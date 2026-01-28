# ml/policy/solver_action_mapping.py
from __future__ import annotations

import re
from typing import Dict, List, Literal, Tuple

from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB

OOPRootKind = Literal["donk", "bet"]

# ----------------------------
# Root kind (OOP at flop root)
# ----------------------------
def oop_root_kind_for_bet_sizing_id(bet_sizing_id: str) -> OOPRootKind:
    """
    Decide whether the OOP root action (at flop root) should be treated as a "donk" node or "bet" node.

    This is not poker-theory; it's purely for solver tree interpretation:
      - If OOP is the preflop caller, their first bet is a donk.
      - If OOP is the preflop aggressor, their first bet is a bet.

    Based on your menu IDs.
    """
    m = (bet_sizing_id or "").strip()

    # Limped pots: OOP is effectively caller at flop root in your setup => donk
    if m.startswith("limped_"):
        return "donk"

    # SRP HU:
    # - srp_hu.PFR_IP  => IP was PFR, OOP is caller => donk
    # - srp_hu.Caller_OOP => OOP explicitly caller => donk
    if m.startswith("srp_hu."):
        if m.endswith("PFR_IP"):
            return "donk"
        if m.endswith("Caller_OOP"):
            return "donk"
        # If you later add PFR_OOP / Caller_IP etc, handle them explicitly:
        if m.endswith("PFR_OOP"):
            return "bet"
        if m.endswith("Caller_IP"):
            return "bet"  # OOP is PFR in that naming
        # Conservative default in SRP: treat OOP as caller => donk
        return "donk"

    # 3bet HU:
    if m.startswith("3bet_hu."):
        if m.endswith("Aggressor_OOP"):
            return "bet"
        if m.endswith("Aggressor_IP"):
            return "donk"
        # conservative default
        return "donk"

    # 4bet HU:
    if m.startswith("4bet_hu."):
        if m.endswith("Aggressor_OOP"):
            return "bet"
        if m.endswith("Aggressor_IP"):
            return "donk"
        return "donk"

    # Unknown menu => conservative: treat OOP as caller => donk
    return "donk"


# ----------------------------
# Tokenization helpers
# ----------------------------
_NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")

def _last_number(s: str) -> float | None:
    m = _NUM.findall(str(s))
    if not m:
        return None
    try:
        return float(m[-1])
    except Exception:
        return None

def _norm_key(k: str) -> str:
    return str(k or "").strip().upper().replace("-", "").replace("_", "").replace(" ", "")

def _is_allin_label(k: str) -> bool:
    u = str(k or "").upper()
    return any(x in u for x in ("ALLIN", "ALL-IN", "ALL IN", "JAM", "SHOVE"))

def _is_check_label(k: str) -> bool:
    return _norm_key(k).startswith("CHECK")

def _is_fold_label(k: str) -> bool:
    return _norm_key(k).startswith("FOLD")

def _is_call_label(k: str) -> bool:
    return _norm_key(k).startswith("CALL")

def _looks_raise_label(k: str) -> bool:
    u = str(k or "").upper()
    return any(x in u for x in ("RAISE", "RERAISE", "RE-RAISE", "3BET", "4BET", "X"))

def _looks_bet_label(k: str) -> bool:
    u = str(k or "").upper()
    return any(x in u for x in ("BET", "DONK", "PROBE"))


# ----------------------------
# Root mapping (CHECK + one size)
# ----------------------------
# These are the only sizes you said are legal for ROOT_ACTION_VOCAB
_ROOT_SIZES = [25, 33, 50, 66, 75, 125]

def _nearest_root_size(pct: int) -> int:
    # ties go to smaller size
    return min(_ROOT_SIZES, key=lambda x: (abs(x - pct), x))

def _root_bet_token_for_size_pct(size_pct: int) -> str:
    """
    Convert an arbitrary size_pct (e.g. 67) into a ROOT vocab token (e.g. BET_66).
    """
    s = int(size_pct)
    s = max(1, min(200, s))
    s = _nearest_root_size(s)
    tok = f"BET_{s}"
    # Safety: vocab might not include BET_125 in some variants; fall back to nearest present
    if tok not in set(ROOT_ACTION_VOCAB):
        # fallback to nearest existing BET_*
        candidates = [t for t in ROOT_ACTION_VOCAB if t.startswith("BET_")]
        if candidates:
            return candidates[-1]
    return tok

def _parse_bet_pct_from_label(label: str) -> int | None:
    """
    Accepts things like:
      'BET 33%' / 'DONK 33%' / 'BET_33' / 'DONK_33' / 'BET33' / 'DONK33'
      or numeric bb labels (rare): 'BET 5.0' (we cannot convert to pct safely here)
    Returns pct if parseable, else None.
    """
    u = str(label or "").upper()

    # Common: explicit percent
    if "%" in u:
        v = _last_number(u)
        if v is None:
            return None
        return int(round(v))

    # Common: BET_33 or DONK_33 etc
    m = re.search(r"(?:BET|DONK|PROBE)[^\d]*([0-9]{1,3})", u.replace("_", " "))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    return None

def map_root_mix_to_root_vocab(
    mix: Dict[str, float],
    *,
    root_kind: OOPRootKind,
    size_pct: int,
) -> Dict[str, float]:
    """
    Map solver root mix -> ROOT_ACTION_VOCAB.

    We keep:
      - CHECK (if present)
      - BET_{size_pct bucketed to {25,33,50,66,75,125}}

    Notes:
    - We do not try to interpret multiple bet sizes at root here.
      Your job submission is size-specific and should yield a single meaningful size.
    - root_kind is accepted for future-proofing. Mapping emits BET_* tokens regardless
      (your vocab doesn't have DONK_*).
    """
    out = {a: 0.0 for a in ROOT_ACTION_VOCAB}

    if not mix:
        return out

    bet_tok = _root_bet_token_for_size_pct(int(size_pct))

    check_mass = 0.0
    bet_mass = 0.0

    for k, p in mix.items():
        if p is None:
            continue
        try:
            prob = float(p)
        except Exception:
            continue
        if prob <= 0:
            continue

        if _is_check_label(k):
            check_mass += prob
            continue

        # treat any bet-like label as “bet”. if the label includes a pct, only
        # accept it if it matches this job’s size (within bucketing tolerance).
        if _looks_bet_label(k):
            pct = _parse_bet_pct_from_label(k)
            if pct is None:
                # unknown bet label -> count it (conservative)
                bet_mass += prob
            else:
                # only count if it buckets to the same vocab token
                if _root_bet_token_for_size_pct(pct) == bet_tok:
                    bet_mass += prob

    # if solver gave no explicit bet mass but did give check, keep check only
    out["CHECK"] = float(check_mass)
    if bet_tok in out:
        out[bet_tok] = float(bet_mass)

    # normalize (if any mass)
    s = sum(out.values())
    if s > 1e-12:
        out = {k: v / s for k, v in out.items()}
    return out


# ----------------------------
# Facing mapping (FOLD/CALL/RAISE_TO_x/ALLIN)
# ----------------------------
# We’ll map raises to the nearest of raise_mults (e.g. [2.0,3.0,4.5]) and then emit:
#   2.0  -> RAISE_TO_200
#   3.0  -> RAISE_TO_300
#   4.5  -> RAISE_TO_450
#
# We DO NOT attempt to reconstruct raise sizes in bb/pot without a faced bet baseline.
# The extractor should already produce “raise-ish” labels. We parse multipliers if present.

def _raise_tok_for_mult(mult: float) -> str:
    # Format to 200/300/450 token style expected by your vocab
    pct = int(round(float(mult) * 100.0))
    return f"RAISE_TO_{pct}"

def _nearest_mult(x: float, mults: List[float]) -> float:
    return min(mults, key=lambda m: (abs(m - x), m))

def _parse_raise_mult_from_label(label: str) -> float | None:
    """
    Accepts:
      - 'RAISE_TO_300', 'RAISETO300'
      - 'RAISE 3X', 'RERAISE 4.5x'
      - 'RAISE_TO 3.0x'
    Returns raise multiple (e.g. 3.0, 4.5) if parseable, else None.
    """
    u = str(label or "").upper()

    # explicit token with 200/300/450 etc
    m = re.search(r"RAISE(?:TO)?[_\s]*([0-9]{3})", u.replace("-", "").replace(" ", ""))
    if m:
        try:
            return float(int(m.group(1)) / 100.0)
        except Exception:
            pass

    # multiplier form: "3x", "4.5x"
    if "X" in u:
        v = _last_number(u)
        if v is not None:
            return float(v)

    return None

def map_facing_mix_to_facing_vocab(
    mix: Dict[str, float],
    *,
    raise_mults: List[float],
) -> Dict[str, float]:
    """
    Map solver facing mix -> FACING_ACTION_VOCAB.

    Keeps:
      - FOLD
      - CALL
      - RAISE_TO_{bucketed mult*100} where mult is nearest in raise_mults
      - ALLIN

    Any unrecognized labels are ignored (not guessed).
    """
    out = {a: 0.0 for a in FACING_ACTION_VOCAB}
    if not mix:
        return out

    mults = [float(x) for x in raise_mults] if raise_mults else [2.0, 3.0, 4.5]

    for k, p in mix.items():
        if p is None:
            continue
        try:
            prob = float(p)
        except Exception:
            continue
        if prob <= 0:
            continue

        if _is_fold_label(k):
            out["FOLD"] += prob
            continue
        if _is_call_label(k):
            out["CALL"] += prob
            continue
        if _is_allin_label(k):
            # allow both 'ALLIN' and 'ALL-IN' etc
            if "ALLIN" in out:
                out["ALLIN"] += prob
            continue

        if _looks_raise_label(k):
            rm = _parse_raise_mult_from_label(k)
            if rm is None:
                # cannot bucket without a multiplier => ignore
                continue
            nearest = _nearest_mult(rm, mults)
            tok = _raise_tok_for_mult(nearest)
            if tok in out:
                out[tok] += prob
            else:
                # if vocab differs, fall back to closest existing RAISE_TO_*
                raise_toks = [t for t in out.keys() if t.startswith("RAISE_TO_")]
                if raise_toks:
                    out[raise_toks[-1]] += prob
            continue

        # ignore unknowns

    # normalize
    s = sum(out.values())
    if s > 1e-12:
        out = {k: v / s for k, v in out.items()}
    return out
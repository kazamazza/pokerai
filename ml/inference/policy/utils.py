from typing import Any, Dict, List, Tuple, Optional, Sequence
import math
import numpy as np
import torch

# ------------------ numeric & safety ------------------
def safe(x: float, lo: float = 1e-9, hi: float = 1 - 1e-9) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))


# ------------------ legality ------------------
def is_legal(action: str, req: Dict[str, Any]) -> bool:
    """
    Very light legality: disallow CHECK if facing a bet.
    Extend later with pot/stack checks.
    """
    facing_bet = bool(req.get("facing_bet", False))
    up = str(action).upper()
    if facing_bet and up == "CHECK":
        return False
    return True


def apply_legality_mask(
        actions: List[str], probs: List[float], req: Dict[str, Any]
) -> Tuple[List[str], List[float]]:
    pairs = [(a, p) for a, p in zip(actions, probs) if is_legal(a, req)]
    if not pairs:
        return ["FOLD"], [1.0]
    acts, ps = zip(*pairs)
    s = sum(ps)
    if s <= 0:
        n = len(ps)
        return list(acts), [1.0 / n] * n
    return list(acts), [float(x / s) for x in ps]


# ------------------ EV / sizing ------------------
def size_map(up: str, pot_bb: float, stack_bb: float) -> float:
    """
    Convert an action like BET_33, RAISE_200, ALLIN into a numeric chip investment.
    """
    try:
        if up.startswith("BET_") or up.startswith("DONK_"):
            return float(up.split("_")[1]) / 100.0 * pot_bb
        if up.startswith("RAISE_"):
            mult = float(up.split("_")[1]) / 100.0
            return mult * pot_bb
        if up == "ALLIN":
            return max(0.0, float(stack_bb))
    except Exception:
        pass
    return 0.0


def ev_one(
        action: str,
        pot_bb: float,
        stack_bb: float,
        eq: Dict[str, float],
        opp: Dict[str, float],
) -> float:
    """
    Compute a naive EV for one action given hero equity and opponent mix.
    """
    eq_win = float(eq.get("p_win", 0.5))
    eq_tie = float(eq.get("p_tie", 0.0))
    e = safe(eq_win + 0.5 * eq_tie)

    p_fold = safe(opp.get("p_fold", 1 / 3))
    p_call = safe(opp.get("p_call", 1 / 3))
    p_raise = safe(opp.get("p_raise", 1 / 3))

    up = str(action).upper()

    if up in ("FOLD", "CHECK"):
        return 0.0
    if up == "CALL":
        return 0.0

    invest = min(size_map(up, pot_bb, stack_bb), stack_bb)
    final_pot = pot_bb + invest + invest
    ev_call = e * final_pot - (1 - e) * invest
    return p_fold * pot_bb + (p_call + p_raise) * ev_call


def guardrails(
        actions: List[str],
        probs: List[float],
        req: Dict[str, Any],
        params: Dict[str, Any] | None = None,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Adjust probability vector to ensure:
      - illegal actions removed
      - probs non-finite -> uniform
      - min floor applied, renormalized
    Returns (actions, probs, notes).
    """
    notes: List[str] = []
    pairs = [(a, p) for a, p in zip(actions, probs) if is_legal(a, req)]
    if not pairs:
        return ["FOLD"], [1.0], ["guardrail: all illegal -> FOLD"]

    acts, ps = zip(*pairs)
    ps = [0.0 if (p is None or not math.isfinite(float(p))) else float(p) for p in ps]

    floor = float((params or {}).get("prob_floor", 1e-6))
    ps = [max(floor, p) for p in ps]
    s = sum(ps) or 1.0
    ps = [float(p / s) for p in ps]

    if any(abs(a - b) > 1e-6 for a, b in zip(ps, probs[:len(ps)])):
        notes.append("guardrail: renormalized/adjusted probabilities")

    return list(acts), ps, notes
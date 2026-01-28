from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


Target = Literal["root", "facing"]


def _norm_size_pct(x: Any) -> Optional[int]:
    """
    Normalize size into int percent-of-pot in [1..200].
    Accepts 0.33/0.66 or 33/66; tolerates strings.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except Exception:
        return None

    # fractions -> pct
    if f <= 3.0:
        f *= 100.0
    # double-scaled guard (3300 -> 33)
    if f > 200.0:
        f /= 100.0

    f = max(1.0, min(200.0, f))
    return int(round(f))


@dataclass(frozen=True)
class RoutedPostflopState:
    target: Target
    # For root: size_pct is required
    size_pct: Optional[int]
    # For facing: faced_size_pct is required
    faced_size_pct: Optional[int]
    # Pass through the original state (normalized sizes may be overwritten)
    state: Dict[str, Any]


def route_postflop_state(state: Dict[str, Any]) -> RoutedPostflopState:
    """
    Decide whether we are at:
      - root (no bet faced): predict CHECK/BET_*
      - facing (bet faced): predict FOLD/CALL/RAISE_TO_*/ALLIN

    Heuristics (any of these => facing):
      - state["facing_bet"] == 1
      - state["facing"] is truthy
      - state has "faced_size_pct" (or "faced_size")
      - state has "facing_bet_bb" / "facing_to_bb" / "facing_bet"
    """
    s = dict(state or {})

    facing_flag = bool(s.get("facing")) or (int(s.get("facing_bet") or 0) == 1)
    if "faced_size_pct" in s or "faced_size" in s:
        facing_flag = True
    if any(k in s for k in ("facing_bet_bb", "facing_to_bb", "facing_bet")):
        facing_flag = True

    if facing_flag:
        faced = _norm_size_pct(s.get("faced_size_pct", s.get("faced_size")))
        # allow "size_pct" to stand in if caller only provided one field
        if faced is None:
            faced = _norm_size_pct(s.get("size_pct"))
        s["faced_size_pct"] = faced
        return RoutedPostflopState(target="facing", size_pct=None, faced_size_pct=faced, state=s)

    # root
    size = _norm_size_pct(s.get("size_pct"))
    s["size_pct"] = size
    return RoutedPostflopState(target="root", size_pct=size, faced_size_pct=None, state=s)
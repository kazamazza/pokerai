from typing import Dict, List, Tuple, Optional, Literal

# -------- Betting menus (unchanged bets; adjusted raises) --------
# Aggressors get two bet sizes; OOP caller may donk one size.
BET_SIZE_MENUS: Dict[str, List[float]] = {
    # SRP
    "srp_hu.PFR_IP":      [0.33, 0.66],
    "srp_hu.PFR_OOP":     [0.33, 0.66],
    "srp_hu.Caller_IP":   [0.33, 0.66],
    "srp_hu.Caller_OOP":  [0.33],        # donk single size

    # 3-bet HU
    "3bet_hu.Aggressor_IP":  [0.33, 0.66],
    "3bet_hu.Aggressor_OOP": [0.33, 0.66],
    "3bet_hu.Caller_IP":     [0.33],
    "3bet_hu.Caller_OOP":    [0.33],     # donk single size if OOP caller

    # 4-bet HU (keep tiny)
    "4bet_hu.Aggressor_IP":  [0.33],
    "4bet_hu.Aggressor_OOP": [0.33],
    "4bet_hu.Caller_IP":     [0.33],
    "4bet_hu.Caller_OOP":    [0.33],

    # Limped
    "limped_single.SB_IP": [0.33],       # SB limped HU: OOP may donk (single size)
    "limped_multi.Any":    [0.33],       # symmetric, no donk key
}

DEFAULT_MENU = [0.33]

# -------- Critical: legal flop raise ladder --------
# Interpret as *raise-to multipliers of the current bet* in percent.
# 150 = 1.5x bet, 200 = 2.0x bet, 300 = 3.0x bet — aligns with training buckets.
RAISE_FLOP_MULT = [150, 200, 300]

# Allow all-in branches on flop to prevent solver pruning away raises when stacks are shallow.
ENABLE_FLOP_ALLIN = True

def _pct_list(xs: List[float]) -> List[int]:
    """e.g. 0.33 -> 33"""
    return sorted({int(round(x * 100)) for x in xs})

def _parse_menu_id(menu_id: str) -> Tuple[str, str]:
    key = (menu_id or "").strip()
    if "." in key:
        a, b = key.split(".", 1)
        return a, b
    return key or "unknown", "unknown"

def _is_aggressor(role: str) -> bool:
    return role.startswith("PFR_") or role.startswith("Aggressor_")

def _is_oop(role: str) -> bool:
    return role.endswith("_OOP") or role == "OOP"

def _make_flop_side(role: str, sizes_pct: List[int], *, allow_donk_for_oop_caller: bool) -> dict:
    """
    Why: ensure raises are present and legal on flop for both roles.
    - 'raise' uses 150/200/300 (bet-relative).
    - enable 'allin' on flop to keep branch viable at short stacks.
    """
    side: Dict[str, object] = {"raise": RAISE_FLOP_MULT, "allin": ENABLE_FLOP_ALLIN}
    if (not _is_aggressor(role)) and _is_oop(role) and allow_donk_for_oop_caller:
        side["donk"] = sizes_pct                      # OOP caller may donk after IP check
    else:
        side["bet"] = sizes_pct                       # default bet family
    return side

def build_contextual_bet_sizes(menu_id: Optional[str]) -> dict:
    """
    Flop-only tree with legal raise ladder and flop all-in enabled.
    Turn/river remain disabled to keep solves lean.
    Returned structure matches build_command_text() expectations.
    """
    key = (menu_id or "").strip()
    group, role = _parse_menu_id(key)

    sizes_pct = _pct_list(BET_SIZE_MENUS.get(key, DEFAULT_MENU))
    allow_donk = role.endswith("Caller_OOP") or role == "Caller_OOP"

    flop_cfg = {
        "ip":  _make_flop_side(role if role.endswith("_IP")  else "Neutral_IP",
                               sizes_pct, allow_donk_for_oop_caller=allow_donk),
        "oop": _make_flop_side(role if role.endswith("_OOP") else "Neutral_OOP",
                               sizes_pct, allow_donk_for_oop_caller=allow_donk),
    }

    # Limped multi: symmetric 'bet'; no donk key.
    if group.startswith("limped_multi"):
        flop_cfg["oop"].pop("donk", None)
        flop_cfg["oop"].setdefault("bet", sizes_pct)

    TURN_RIVER_DISABLED = {"ip": {"bet": []}, "oop": {"bet": []}}  # keep small trees

    return {
        "flop": flop_cfg,
        "turn":  TURN_RIVER_DISABLED,
        "river": TURN_RIVER_DISABLED,
    }
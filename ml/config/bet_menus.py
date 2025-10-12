from typing import Dict, List, Tuple, Optional, Literal

# -------- Betting menus (unchanged bets; adjusted raises) --------
# Aggressors get two bet sizes; OOP caller may donk one size.

BET_SIZE_MENUS_NL10 = {
    # SRP (single raised)
    "srp_hu.PFR_IP":      [0.33, 0.66],     # 1/3, 2/3 → range bet + polarizing size
    "srp_hu.Caller_OOP":  [0.33],           # single donk size

    # 3-bet pots
    "3bet_hu.Aggressor_IP":  [0.25, 0.33, 0.66],  # small-pressure spectrum; drop pot/0.75
    "3bet_hu.Aggressor_OOP": [0.33, 0.75],        # allow larger OOP size

    # 4-bet pots
    "4bet_hu.Aggressor_IP":  [0.33],        # keep small, shallow SPR
    "4bet_hu.Aggressor_OOP": [0.33],        # same

    # Limped pots
    "limped_single.SB_IP": [0.33],
    "limped_multi.Any":    [0.33],
}

DEFAULT_MENU_NL10 = [0.33]
RAISE_FLOP_MULT_NL10 = [150, 200, 300]
ENABLE_FLOP_ALLIN_NL10 = True


BET_SIZE_MENUS_NL25 = {
    # SRP
    "srp_hu.PFR_IP":      [0.25, 0.33, 0.66],     # add tiny stab for nut/range crush boards
    "srp_hu.Caller_OOP":  [0.33],                 # keep donk simple

    # 3-bet HU
    "3bet_hu.Aggressor_IP":  [0.25, 0.33, 0.66],  # remove pot; emphasize small-to-mid
    "3bet_hu.Aggressor_OOP": [0.33, 0.75],        # give OOP a bigger lever

    # 4-bet HU
    "4bet_hu.Aggressor_IP":  [0.33],              # keep small, shallow SPR
    "4bet_hu.Aggressor_OOP": [0.33],

    # Limped
    "limped_single.SB_IP": [0.33],
    "limped_multi.Any":    [0.33],
}
DEFAULT_MENU_NL25 = [0.33]
RAISE_FLOP_MULT_NL25 = [150, 200, 300, 400]  # retain 4x ladder for polarized spots
ENABLE_FLOP_ALLIN_NL25 = True

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

def _legalize_raise_mult(raise_mult: list[int]) -> list[int]:
    # keep only >100% to ensure a legal raise-to
    out = sorted({int(m) for m in raise_mult if m > 100})
    # fallback to safe defaults if user supplied nonsense
    return out or [150, 200, 300]

def _make_flop_side(
    role: str,
    sizes_pct: list[float],
    *,
    allow_donk_for_oop_caller: bool,
    raise_mult: list[int],
    enable_allin: bool,
) -> dict:
    """
    Ensure raises are legal on flop for both roles.
    - 'raise' uses stake-specific raise ladder (e.g. 150/200/300/400)
    - enable 'allin' on flop to keep branch viable at short stacks.
    """
    side: dict[str, object] = {
        "raise": _legalize_raise_mult(raise_mult),
        "allin": enable_allin,
    }
    if (not _is_aggressor(role)) and _is_oop(role) and allow_donk_for_oop_caller:
        side["donk"] = sizes_pct
    else:
        side["bet"] = sizes_pct
    return side


def build_contextual_bet_sizes(menu_id: Optional[str], *, stakes: str = "NL10") -> dict:
    """
    Flop-only tree with legal raise ladder and flop all-in enabled.
    Stake parameter selects correct menus/raise ladders.
    """
    key = (menu_id or "").strip()
    group, role = _parse_menu_id(key)

    # --- stake routing ---
    if stakes.upper() == "NL25":
        BETS = BET_SIZE_MENUS_NL25
        DEFAULT = DEFAULT_MENU_NL25
        RAISE_MULT = RAISE_FLOP_MULT_NL25
        ENABLE_ALLIN = ENABLE_FLOP_ALLIN_NL25
    else:  # fallback NL10
        BETS = BET_SIZE_MENUS_NL10
        DEFAULT = DEFAULT_MENU_NL10
        RAISE_MULT = RAISE_FLOP_MULT_NL10
        ENABLE_ALLIN = ENABLE_FLOP_ALLIN_NL10

    sizes_pct = _pct_list(BETS.get(key, DEFAULT))
    allow_donk = role.endswith("Caller_OOP") or role == "Caller_OOP"

    flop_cfg = {
        "ip": _make_flop_side(
            role if role.endswith("_IP") else "Neutral_IP",
            sizes_pct,
            allow_donk_for_oop_caller=allow_donk,
            raise_mult=RAISE_MULT,
            enable_allin=ENABLE_ALLIN,
        ),
        "oop": _make_flop_side(
            role if role.endswith("_OOP") else "Neutral_OOP",
            sizes_pct,
            allow_donk_for_oop_caller=allow_donk,
            raise_mult=RAISE_MULT,
            enable_allin=ENABLE_ALLIN,
        ),
    }

    # Limped multi: symmetric bet (no donk)
    if group.startswith("limped_multi"):
        flop_cfg["oop"].pop("donk", None)
        flop_cfg["oop"].setdefault("bet", sizes_pct)

    TURN_RIVER_DISABLED = {"ip": {"bet": []}, "oop": {"bet": []}}

    return {
        "flop": flop_cfg,
        "turn": TURN_RIVER_DISABLED,
        "river": TURN_RIVER_DISABLED,
    }
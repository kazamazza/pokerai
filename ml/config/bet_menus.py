from typing import Dict, List, Tuple, Optional, Literal
from ml.config.solver import STAKE_CFG
from ml.core.types import Stakes


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

def _legalize_raise_mult(mult: List[float]) -> List[float]:
    """Normalize/validate raise multipliers."""
    vals = [float(x) for x in mult if x > 1.0]
    return sorted(set(vals))


def _make_flop_side(
    role: str,
    sizes_frac: List[float],
    *,
    allow_donk_for_oop_caller: bool,
    raise_mult: List[float],
    enable_allin: bool,
) -> Dict[str, object]:
    """
    Build side-specific betting config.
    - bets: fractional pot sizes (e.g. [0.33, 0.66])
    - raises: absolute multipliers (e.g. [1.5, 2.0, 3.0])
    """
    side: Dict[str, object] = {
        "raise": _legalize_raise_mult(raise_mult),  # e.g. [1.5, 2.0, 3.0]
        "allin": enable_allin,
    }

    if (not _is_aggressor(role)) and _is_oop(role) and allow_donk_for_oop_caller:
        side["donk"] = sizes_frac
    else:
        side["bet"] = sizes_frac

    return side


def build_contextual_bet_sizes(menu_id: Optional[str], *, stake: Stakes = Stakes.NL10) -> dict:
    """
    Build flop/turn/river bet-size structure for a given menu ID and stake.
    Uses the unified STAKE_CFG for all stake-dependent parameters.
    """
    key = (menu_id or "").strip()
    group, role = _parse_menu_id(key)

    # --- load from canonical config
    cfg = STAKE_CFG.get(stake, STAKE_CFG[Stakes.NL10])
    BETS = cfg["bet_menus"]
    DEFAULT = next(iter(cfg["bet_menus"].values()))  # use first entry as fallback
    RAISE_MULT = cfg["raise_mult"]
    ENABLE_ALLIN = cfg.get("flop_allin", True)

    # --- pick bet sizes for this menu
    sizes_frac: List[float] = list(BETS.get(key, DEFAULT))
    allow_donk = role.endswith("Caller_OOP") or role == "Caller_OOP"

    flop_cfg = {
        "ip": _make_flop_side(
            role if role.endswith("_IP") else "Neutral_IP",
            sizes_frac,
            allow_donk_for_oop_caller=allow_donk,
            raise_mult=RAISE_MULT,
            enable_allin=ENABLE_ALLIN,
        ),
        "oop": _make_flop_side(
            role if role.endswith("_OOP") else "Neutral_OOP",
            sizes_frac,
            allow_donk_for_oop_caller=allow_donk,
            raise_mult=RAISE_MULT,
            enable_allin=ENABLE_ALLIN,
        ),
    }

    # Limped multi: symmetric, no donk
    if group.startswith("limped_multi"):
        flop_cfg["oop"].pop("donk", None)
        flop_cfg["oop"].setdefault("bet", sizes_frac)

    TURN_RIVER_DISABLED = {"ip": {"bet": []}, "oop": {"bet": []}}

    return {"flop": flop_cfg, "turn": TURN_RIVER_DISABLED, "river": TURN_RIVER_DISABLED}
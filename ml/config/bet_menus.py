# --- FLOP-ONLY MENUS FOR ML (lean, multi-action, bounded tree) ---

from typing import Optional, Dict, List, Tuple

# Aggressors get two bet sizes; caller OOP may donk one size.
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

DEFAULT_MENU = [0.33]  # fallback

# Single flop raise ladder to allow a raise branch without explosion
RAISE_FLOP = [66]      # % pot
# Turn/River disabled (keeps solve small & fast)
TURN_RIVER_DISABLED = {"ip": {"bet": []}, "oop": {"bet": []}}

def _pct_list(xs):  # 0.33 -> 33
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
    side = {"raise": RAISE_FLOP, "allin": False}   # one raise; no allin for flop-only v1
    if (not _is_aggressor(role)) and _is_oop(role) and allow_donk_for_oop_caller:
        side["donk"] = sizes_pct                    # OOP caller may donk
    else:
        side["bet"] = sizes_pct                     # default bet family
    return side

def build_contextual_bet_sizes(menu_id: Optional[str]) -> dict:
    """
    Flop-only tree. Turn/river intentionally empty to avoid deeper streets.
    Returns:
      {
        "flop": {
          "ip":  {"bet":[..] or "donk":[..], "raise":[66], "allin":False},
          "oop": {...}
        },
        "turn":  {"ip":{"bet":[]}, "oop":{"bet":[]}},
        "river": {"ip":{"bet":[]}, "oop":{"bet":[]}},
      }
    """
    key = (menu_id or "").strip()
    group, role = _parse_menu_id(key)

    sizes_pct = _pct_list(BET_SIZE_MENUS.get(key, DEFAULT_MENU))
    allow_donk = role.endswith("Caller_OOP") or role == "Caller_OOP"

    flop = {
        "ip":  _make_flop_side(role if role.endswith("_IP")  else "Neutral_IP",
                               sizes_pct, allow_donk_for_oop_caller=allow_donk),
        "oop": _make_flop_side(role if role.endswith("_OOP") else "Neutral_OOP",
                               sizes_pct, allow_donk_for_oop_caller=allow_donk),
    }

    # Limped multi: forbid 'donk' explicitly; symmetric 'bet' only
    if group.startswith("limped_multi"):
        flop["oop"].pop("donk", None)
        flop["oop"].setdefault("bet", sizes_pct)

    return {
        "flop": flop,
        "turn":  TURN_RIVER_DISABLED,
        "river": TURN_RIVER_DISABLED,
    }
# --- replace the old BET_MENUS/_format_bet_sizes section with this ---
from typing import Optional

# Your per-spot flop size lists (fractions of pot)
BET_SIZE_MENUS = {
    # SRP
    "srp_hu.PFR_IP":     [0.25, 0.33, 0.50, 0.75, 1.00],
    "srp_hu.PFR_OOP":    [0.33, 0.50, 0.75, 1.00],
    "srp_hu.Caller_IP":  [0.25, 0.50, 0.75],
    "srp_hu.Caller_OOP": [0.25, 0.50, 0.75],  # donk (non-agg OOP) allowed
    # 3-bet
    "3bet_hu.Aggressor_IP":  [0.25, 0.50, 0.75],
    "3bet_hu.Aggressor_OOP": [0.25, 0.50, 0.75],
    "3bet_hu.Caller_IP":     [0.33, 0.66, 1.00],
    "3bet_hu.Caller_OOP":    [0.33, 0.66, 1.00],  # donk allowed (non-agg OOP)
    # 4-bet
    "4bet_hu.Aggressor_IP":  [0.25, 0.50, 0.75],
    "4bet_hu.Aggressor_OOP": [0.25, 0.50, 0.75],
    "4bet_hu.Caller_IP":     [0.33, 0.66, 1.00],
    "4bet_hu.Caller_OOP":    [0.33, 0.66, 1.00],
    # Limped
    "limped_single.SB_IP": [0.33, 0.66, 1.00],   # generic limp HU
    "limped_multi.Any":    [0.33, 0.66, 1.00],   # generic limp multi
}
DEFAULT_MENU = [0.33, 0.66, 1.00]  # if unknown id

TURN_RIVER_DEFAULT = [0.50, 0.66, 1.00]  # compact & stable across contexts
RAISE_DEFAULT = [66, 100, 150]           # % of last bet (typical raise ladder)

def _pct_list(xs):  # 0.33 -> 33
    return sorted({int(round(x * 100)) for x in xs})

def _parse_menu_id(menu_id: str) -> tuple[str, str]:  # ("srp_hu", "PFR_IP")
    menu_id = (menu_id or "").strip()
    if "." in menu_id:
        a, b = menu_id.split(".", 1)
        return a, b
    return "unknown", "unknown"

def _is_aggressor(role: str) -> bool:
    return any(tag in role for tag in ("PFR_", "Aggressor_"))

def _is_oop(role: str) -> bool:
    return role.endswith("_OOP") or role.endswith(".OOP") or role == "OOP"

def _make_flop_side(role: str, sizes_pct: list[int]) -> dict:
    """
    For FLOP only:
      - Aggressor (c-better): use 'bet'
      - Non-aggressor OOP: allow 'donk'
      - Non-aggressor IP: use 'bet' (bet if checked to)
    """
    side = {"raise": RAISE_DEFAULT, "allin": True}
    if (not _is_aggressor(role)) and _is_oop(role):
        side["donk"] = sizes_pct
    else:
        side["bet"] = sizes_pct
    return side

def build_contextual_bet_sizes(menu_id: Optional[str]) -> dict:
    """
    Returns the full per-street structure expected by build_command_text:
      { "flop": {"ip": {...}, "oop": {...}}, "turn": {...}, "river": {...} }
    """
    key = (menu_id or "").strip()
    sizes = BET_SIZE_MENUS.get(key, DEFAULT_MENU)
    sizes_pct = _pct_list(sizes)
    turn_pct = _pct_list(TURN_RIVER_DEFAULT)
    river_pct = _pct_list(TURN_RIVER_DEFAULT)

    # Determine roles for flop sides from the menu_id
    group, role = _parse_menu_id(key)
    # If unknown, assume neutral roles so both sides can bet (no OOP donk)
    ip_role = role if role.endswith("_IP") else "Neutral_IP"
    oop_role = role if role.endswith("_OOP") else "Neutral_OOP"

    bet_sizes = {
        "flop": {
            "ip":  _make_flop_side(ip_role, sizes_pct),
            "oop": _make_flop_side(oop_role, sizes_pct),
        },
        "turn": {
            "ip":  {"bet": turn_pct, "raise": RAISE_DEFAULT, "allin": True},
            "oop": {"bet": turn_pct, "raise": RAISE_DEFAULT, "allin": True},
        },
        "river": {
            "ip":  {"bet": river_pct, "raise": RAISE_DEFAULT, "allin": True},
            "oop": {"bet": river_pct, "raise": RAISE_DEFAULT, "allin": True},
        },
    }

    # Limped multi: we don’t want accidental OOP “donk” semantics—just bets for both sides
    if group.startswith("limped_multi"):
        bet_sizes["flop"]["oop"].pop("donk", None)
        if "bet" not in bet_sizes["flop"]["oop"]:
            bet_sizes["flop"]["oop"]["bet"] = sizes_pct

    return bet_sizes
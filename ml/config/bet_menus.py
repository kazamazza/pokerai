# --- FLOP-ONLY, SMALL TREE MENUS ---

from typing import Optional

# We keep a single compact list for aggressors (c-bet) and a single size for OOP caller (donk).
# Context still flows through <group>.<role> but sizes are tiny to cap branching.

BET_SIZE_MENUS = {
    # SRP
    "srp_hu.PFR_IP":      [0.33, 0.66],   # c-bet ladder (2 sizes)
    "srp_hu.PFR_OOP":     [0.33, 0.66],
    "srp_hu.Caller_IP":   [0.33, 0.66],
    "srp_hu.Caller_OOP":  [0.33],         # OOP caller donk = single size

    # 3-bet HU
    "3bet_hu.Aggressor_IP":  [0.33, 0.66],
    "3bet_hu.Aggressor_OOP": [0.33, 0.66],
    "3bet_hu.Caller_IP":     [0.33, 0.66],
    "3bet_hu.Caller_OOP":    [0.33],

    # 4-bet HU (keep same—trees are tiny anyway)
    "4bet_hu.Aggressor_IP":  [0.33],
    "4bet_hu.Aggressor_OOP": [0.33],
    "4bet_hu.Caller_IP":     [0.33],
    "4bet_hu.Caller_OOP":    [0.33],

    # Limped
    "limped_single.SB_IP": [0.33],
    "limped_multi.Any":    [0.33],   # plain bet both sides (no 'donk' key)
}

DEFAULT_MENU = [0.33]  # fallback

def _pct_list(xs):  # 0.33 -> 33
    return sorted({int(round(x * 100)) for x in xs})

def _parse_menu_id(menu_id: str) -> tuple[str, str]:
    key = (menu_id or "").strip()
    if "." in key:
        a, b = key.split(".", 1)
        return a, b
    return key or "unknown", "unknown"

def _is_aggressor(role: str) -> bool:
    return role.startswith("PFR_") or role.startswith("Aggressor_")

def _is_oop(role: str) -> bool:
    return role.endswith("_OOP") or role == "OOP"

def _make_flop_side(role: str, sizes_pct: list[int], *, allow_donk_for_oop_caller: bool) -> dict:
    """
    Flop only:
      - Aggressor: 'bet' with 2 sizes (33,66) typically
      - Non-agg OOP: 'donk' with 1 size (33) if allowed
      - No flop raises, no flop allin
    """
    side = {"raise": []}  # explicitly no raises on flop
    if (not _is_aggressor(role)) and _is_oop(role) and allow_donk_for_oop_caller:
        side["donk"] = sizes_pct
    else:
        side["bet"] = sizes_pct
    return side

def build_contextual_bet_sizes(menu_id: Optional[str]) -> dict:
    """
    Flop-only tree. Turn/river are intentionally empty to avoid building deeper streets.
    """
    key = (menu_id or "").strip()
    group, role = _parse_menu_id(key)

    sizes_pct = _pct_list(BET_SIZE_MENUS.get(key, DEFAULT_MENU))

    # Allow OOP donk only for explicit caller OOP roles
    allow_donk = role.endswith("Caller_OOP") or role == "Caller_OOP"

    flop = {
        "ip":  _make_flop_side(role if role.endswith("_IP")  else "Neutral_IP",
                               sizes_pct, allow_donk_for_oop_caller=allow_donk),
        "oop": _make_flop_side(role if role.endswith("_OOP") else "Neutral_OOP",
                               sizes_pct, allow_donk_for_oop_caller=allow_donk),
    }

    # Limped multi: force plain bet (no 'donk' key)
    if group.startswith("limped_multi"):
        flop["oop"].pop("donk", None)
        flop["oop"].setdefault("bet", sizes_pct)

    bet_sizes = {
        "flop": flop,
        # Leave turn/river empty so the solver doesn’t build them at all
        "turn":  {"ip": {"bet": []}, "oop": {"bet": []}},
        "river": {"ip": {"bet": []}, "oop": {"bet": []}},
    }
    return bet_sizes
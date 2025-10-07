from typing import Tuple, Optional

from ml.config.bet_menus import BET_SIZE_MENUS
from ml.etl.utils.positions import canon_pos
from ml.etl.rangenet.preflop.monker_manifest import expected_menu_id


# Exact ids your BET_SIZE_MENUS exposes
MENU_TAG_TO_ID = {
    "srp_ip":    "srp_hu.PFR_IP",
    "srp_oop":   "srp_hu.Caller_OOP",  # OOP caller (donk after check line)
    "3bet_ip":   "3bet_hu.Aggressor_IP",
    "3bet_oop":  "3bet_hu.Aggressor_OOP",
    "4bet_ip":   "4bet_hu.Aggressor_IP",
    "4bet_oop":  "4bet_hu.Aggressor_OOP",
    "limp":      "limped_single.SB_IP",
    "limp_multi":"limped_multi.Any",
}

def _ctx_for_lookup(ctx: str) -> str:
    """
    Route SRP-like contexts to 'SRP' since your Monker index is SRP-centric.
    Other ctx pass through (your lookup has built-in fallback).
    """
    c = str(ctx).upper()
    return "SRP" if c in ("VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL", "SRP") else c


def _infer_topology_and_roles(ctx: str, ip: str, oop: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Minimal deterministic inference from (ctx, ip, oop) → (topology, opener, three_bettor).
    We use common poker heuristics that are sufficient to bind menus + pot sizes:

      - SRP: if pairing is late-pos vs blind, opener=late-pos; else opener=ip by default.
      - 3bet: if three-bettor is OOP (common vs BTN), assume OOP; else IP.
      - 4bet: we don’t need opener/3bettor to compute pot, but we return reasonable defaults.
      - Limped: opener='LIMP', three_bettor=None.

    This is for *manifest* construction only; training sees consistent menus and pots.
    """
    c = str(ctx).upper()
    ip, oop = canon_pos(ip), canon_pos(oop)

    if c in ("SRP", "VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL"):
        topo = "srp_hu"
        # late-pos vs blind → late-pos opened
        late = {"BTN", "CO"}
        blinds = {"SB", "BB"}
        if ip in late and oop in blinds:
            opener = ip
        elif oop in late and ip in blinds:
            opener = oop
        else:
            # default: assume IP was the opener
            opener = ip
        return topo, opener, None

    if c == "VS_3BET":
        topo = "3bet_hu"
        # typical pool: blinds 3bet OOP vs BTN/CO IP
        if ip in {"BTN", "CO"} and oop in {"SB", "BB"}:
            opener = ip
            three_bettor = oop  # OOP 3-bettor
        elif oop in {"BTN", "CO"} and ip in {"SB", "BB"}:
            opener = oop
            three_bettor = ip   # OOP 3-bettor
        else:
            # default: three-bettor is the IP
            opener = oop
            three_bettor = ip
        return topo, opener, three_bettor

    if c == "VS_4BET":
        topo = "4bet_hu"
        # default placeholders; menus/pot do not require exact seats here
        # assume 4-bettor is OOP if BTN vs blind, otherwise IP
        if ip in {"BTN", "CO"} and oop in {"SB", "BB"}:
            opener = ip
            three_bettor = oop  # 3-bettor OOP, 4-bettor could be either; not needed for pot/menu
        else:
            opener = oop
            three_bettor = ip
        return topo, opener, three_bettor

    if c == "LIMPED_SINGLE":
        return "limped_single", "LIMP", None

    if c == "LIMPED_MULTI":
        # multiway is not used here (builder is HU), keep a safe label anyway
        return "limped_multi", "LIMP", None

    # default to SRP if unknown
    return "srp_hu", ip, None

# =========================
# Pots: deterministic formulas
# =========================
def _pot_srp(open_x: float) -> float:
    return 1.5 + 2.0 * float(open_x)


def _pot_3bet(final_x: float) -> float:
    return 1.5 + 2.0 * float(final_x)


def _pot_4bet(final_x: float) -> float:
    return 1.5 + 2.0 * float(final_x)


def _compute_pot_bb(
    ctx: str,
    opener: Optional[str],
    ip: str,
    three_bettor: Optional[str],
    *,
    sizes: Mapping[str, Any],
) -> float:
    """
    Compute deterministic starting pot (in BB) from context + roles, using runtime sizes:
      sizes = {
        "open_x":     {"UTG":..., "HJ":..., "CO":..., "BTN":..., "SB":...},
        "threebet_x": {"IP":..., "OOP":...},
        "fourbet_x":  24.0,
      }
    """
    # Resolve size tables with safe defaults
    open_x     = sizes.get("open_x",     {"UTG": 3.0, "HJ": 2.5, "CO": 2.5, "BTN": 2.5, "SB": 3.0})
    threebet_x = sizes.get("threebet_x", {"IP": 7.5, "OOP": 9.0})
    fourbet_x  = float(sizes.get("fourbet_x", 24.0))

    # We only need shape hints from topology inference
    _topo, opener_h, three_h = _infer_topology_and_roles(ctx, ip, "X")
    c = str(ctx).upper()

    if c in ("SRP", "VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL"):
        op_seat = (opener or opener_h or "BTN")
        x = float(open_x.get(op_seat, 2.5))
        return _pot_srp(x)

    if c == "VS_3BET":
        ip_is_three = (three_bettor or three_h) == ip
        final = float(threebet_x["IP"] if ip_is_three else threebet_x["OOP"])
        return _pot_3bet(final)

    if c == "VS_4BET":
        return _pot_4bet(fourbet_x)

    if c == "LIMPED_SINGLE":
        return 1.5

    if c == "LIMPED_MULTI":
        return 1.5  # HU builder won’t use multiway; safe default

    return 6.0


# =========================
# Menu binding (topology+roles)
# =========================
def _menu_for(ctx, ip, oop, opener, three_bettor, menu_tag: str | None = None):
    topo, opener_h, three_h = _infer_topology_and_roles(ctx, ip, oop)

    # 1) explicit tag wins
    if menu_tag:
        tag = str(menu_tag).strip().lower()
        menu_id = MENU_TAG_TO_ID.get(tag)
        if not menu_id:
            raise ValueError(f"Unknown bet_menus tag '{menu_tag}'")
        if menu_id not in BET_SIZE_MENUS:
            raise ValueError(f"Menu id '{menu_id}' from tag '{menu_tag}' not in BET_SIZE_MENUS")
        return menu_id, BET_SIZE_MENUS[menu_id]

    # 2) fall back to inference (unchanged)
    menu_id = expected_menu_id(
        topo=topo,
        ip=canon_pos(ip),
        oop=canon_pos(oop),
        opener=(opener or opener_h),
        three_bettor=(three_bettor or three_h),
    )
    if not menu_id:
        if topo == "limped_multi": menu_id = "limped_multi.Any"
        elif topo == "limped_single": menu_id = "limped_single.SB_IP"
        else:
            raise ValueError(f"Cannot derive bet_sizing_id for ctx={ctx} topo={topo} ip={ip} oop={oop}")
    if menu_id not in BET_SIZE_MENUS:
        raise ValueError(f"Unknown bet_sizing_id '{menu_id}'")
    return menu_id, BET_SIZE_MENUS[menu_id]
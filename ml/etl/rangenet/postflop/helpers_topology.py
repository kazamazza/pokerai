from typing import Tuple, Optional, Mapping, Any, List

from ml.config.solver import STAKE_CFG
from ml.core.types import Stakes
from ml.etl.utils.positions import canon_pos
from ml.etl.rangenet.preflop.monker_manifest import expected_menu_id


MENU_TAG_TO_ID = {
    "srp_ip":    "srp_hu.PFR_IP",
    "srp_oop":   "srp_hu.Caller_OOP",  # OOP caller (donk after check line)
    "3bet_ip":   "3bet_hu.Aggressor_IP",
    "3bet_oop":  "3bet_hu.Aggressor_OOP",
    "4bet_ip":   "4bet_hu.Aggressor_IP",
    "4bet_oop":  "4bet_hu.Aggressor_OOP",
    "limp":      "limped_single.SB_IP",
    "limp_multi":"limped_multi.Any",
    "bvs": "srp_hu.PFR_IP"
}


def _pot_formula(final_x: float, base: float = 1.5) -> float:
    # blinds (SB+BB)=1.5, plus 2 * final raise-to (HU)
    return base + 2.0 * float(final_x)

def compute_pot_bb(*, ctx: str, ip: str, opener: str|None,
                   three_bettor: str|None, stake: Stakes) -> float:
    cfg   = STAKE_CFG[stake]
    openx = cfg["open_x"]; threex = cfg["threebet_x"]; fourx = cfg["fourbet_x"]
    padj  = cfg["pot_adj"]
    c = (ctx or "").upper()

    # limp baselines (HU derivation)
    if c in {"LIMPED_SINGLE","LIMPED_MULTI"}:
        return 1.5

    # SRP
    if c in {"SRP","VS_OPEN","OPEN","VS_OPEN_RFI","BLIND_VS_STEAL"}:
        op = (opener or "BTN").upper()
        x  = float(openx.get(op, 2.5))
        return padj["srp"] * _pot_formula(x)

    # 3bet
    if c == "VS_3BET":
        ip_is_three = (three_bettor or "").upper() == ip.upper()
        x  = float(threex["IP"] if ip_is_three else threex["OOP"])
        return padj["threebet"] * _pot_formula(x)

    # 4bet
    if c == "VS_4BET":
        return padj["fourbet"] * _pot_formula(float(fourx))

    # fallback
    return 6.0

def _ctx_for_lookup(ctx: str) -> str:
    """
    Route SRP-like contexts to 'SRP' since your Monker index is SRP-centric.
    Other ctx pass through (your lookup has built-in fallback).
    """
    c = str(ctx).upper()
    return "SRP" if c in ("VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL", "SRP") else c


def raise_mult_for(stake: Stakes) -> list[float]:
    return STAKE_CFG[stake]["raise_mult"]

def flop_allin_for(stake: Stakes) -> bool:
    return STAKE_CFG[stake]["flop_allin"]


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
# Menu binding (topology+roles)
# =========================
def _default_menu_for(stake: Stakes) -> List[float]:
    return STAKE_CFG[stake]["default_menu"]

def bet_menu_for(menu_id: str, stake: Stakes = Stakes.NL10) -> List[float]:
    """
    Stake-aware bet sizes (fractions of pot) for a given menu id.
    Falls back to stake's default menu.
    """
    table = STAKE_CFG[stake]["bet_menus"]
    return list(table.get(menu_id))

# Lightweight synonyms so YAML can stay clean
_TAG_MAP_STATIC = {
    "srp_ip":     "srp_hu.PFR_IP",
    "srp_oop":    "srp_hu.Caller_OOP",
    "bvs":        "srp_hu.PFR_IP",          # blind-vs-steal treated as SRP PFR_IP
    "3bet_ip":    "3bet_hu.Aggressor_IP",
    "3bet_oop":   "3bet_hu.Aggressor_OOP",
    "4bet_ip":    "4bet_hu.Aggressor_IP",
    "4bet_oop":   "4bet_hu.Aggressor_OOP",
}

def _map_tag_dynamic(tag: str, topo: str, ip: str, oop: str) -> Optional[str]:
    """
    Tags that need context/roles to resolve to a concrete menu_id.
    """
    tag = (tag or "").lower()
    topo = (topo or "").lower()
    ip, oop = (ip or "").upper(), (oop or "").upper()

    if tag == "4bet":
        # default: if OOP is a blind, make OOP aggressor
        return "4bet_hu.Aggressor_OOP" if oop in {"SB", "BB"} else "4bet_hu.Aggressor_IP"

    if tag == "limp":
        if topo == "limped_multi":  return "limped_multi.Any"
        if topo == "limped_single": return "limped_single.SB_IP"
        return "limped_single.SB_IP"

    return None

def _menu_for(ctx, ip, oop, opener, three_bettor,
              menu_tag: Optional[str] = None,
              stake: Stakes = Stakes.NL10) -> Tuple[str, List[float]]:
    """
    Decide the concrete bet_sizing_id (menu_id) and return stake-aware sizes.
    Priority:
      1) explicit tag (static map or dynamic resolution)
      2) inferred by topology/roles (expected_menu_id)
    """
    topo, opener_h, three_h = _infer_topology_and_roles(ctx, ip, oop)

    # --- 1) explicit tag wins ---
    if menu_tag:
        tag = str(menu_tag).strip().lower()
        # static first
        menu_id = _TAG_MAP_STATIC.get(tag)
        # dynamic for special tags
        if not menu_id:
            menu_id = _map_tag_dynamic(tag, topo, canon_pos(ip), canon_pos(oop))
        if not menu_id:
            raise ValueError(f"Unknown bet_menus tag '{menu_tag}'")

        # validate against stake config
        if menu_id not in STAKE_CFG[stake]["bet_menus"]:
            print(f"Unknown bet_menus tag '{menu_tag}'")
            # if not present, still return with default sizes (explicit but rare)
            return menu_id, bet_menu_for(menu_id, stake=stake)
        return menu_id, bet_menu_for(menu_id, stake=stake)

    # --- 2) inferred path (unchanged logic you already use) ---
    menu_id = expected_menu_id(
        topo=topo,
        ip=canon_pos(ip),
        oop=canon_pos(oop),
        opener=(opener or opener_h),
        three_bettor=(three_bettor or three_h),
    )
    if not menu_id:
        if topo == "limped_multi":   menu_id = "limped_multi.Any"
        elif topo == "limped_single": menu_id = "limped_single.SB_IP"
        else:
            raise ValueError(f"Cannot derive bet_sizing_id for ctx={ctx} topo={topo} ip={ip} oop={oop}")

    return menu_id, bet_menu_for(menu_id, stake=stake)
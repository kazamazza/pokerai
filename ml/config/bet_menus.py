from typing import Dict, List, Tuple, Optional, Literal
from ml.config.solver import STAKE_CFG
from ml.core.types import Stakes


# --- unchanged helpers (kept for context) ---
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
    vals = [float(x) for x in mult if x and float(x) > 1.0]
    return sorted(set(vals))

def _make_flop_side(
    role: str,
    sizes_frac: List[float],
    *,
    allow_donk_for_oop_caller: bool,
    raise_mult: List[float],
    enable_allin: bool,
) -> Dict[str, object]:
    side: Dict[str, object] = {
        "raise": _legalize_raise_mult(raise_mult),
        "allin": enable_allin,
    }
    # Donk only when OOP is the caller profile.
    if (not _is_aggressor(role)) and _is_oop(role) and allow_donk_for_oop_caller:
        side["donk"] = sizes_frac
    else:
        side["bet"] = sizes_frac
    return side

# --- desired sizes from STAKE_CFG (unchanged API) ---
def _desired_fracs_for(menu_id: str, stake: Stakes) -> List[float]:
    cfg = STAKE_CFG.get(stake, STAKE_CFG[Stakes.NL10])
    bets = cfg["bet_menus"]
    default = next(iter(bets.values()))
    return list(bets.get((menu_id or "").strip(), default))

# --- UPDATED: stack-aware legalization, but **no new functions/keys** ---
def _legalize_fracs_by_stack(
    desired_fracs: List[float],
    *,
    group: str,
    role: str,
    effective_stack_bb: Optional[float],
    stake: Stakes,
) -> List[float]:
    """
    Convert desired fractions to stack-appropriate fractions that the console will honor.
    Policy is driven by STAKE_CFG['stack_buckets'] when present; else defaults to:
      - short (<=30bb): [0.25, 0.50]
      - medium (<=80bb): [0.33] or [0.33, 0.66] if two-size desired
      - deep (>80bb): [0.33, 0.66] (+1.00 only when desired or allowed)
    Special-cases:
      - 4bet*: always [0.33]
      - limped_single*: [0.33]
    """
    cfg = STAKE_CFG.get(stake, STAKE_CFG[Stakes.NL10])
    g = (group or "").lower()

    # Special-cases first
    if "4bet" in g:
        return [0.33]
    if g.startswith("limped_single"):
        return [0.33]

    # Desired set
    want = sorted(set(float(x) for x in desired_fracs))
    want_two = len(want) >= 2
    want_has_pot = any(abs(x - 1.00) < 1e-6 for x in want)

    # If no stack provided, keep desired as-is
    if effective_stack_bb is None:
        return want

    s = float(effective_stack_bb)

    # Configurable buckets (optional)
    sb = cfg.get("stack_buckets")
    if sb:
        short_max  = float(sb.get("short",  {"max_bb": 30}).get("max_bb", 30))
        short_sz   = [float(x) for x in sb.get("short",  {}).get("sizes", [0.25, 0.50])]
        med_max    = float(sb.get("medium", {"max_bb": 80}).get("max_bb", 80))
        med_sz     = [float(x) for x in sb.get("medium", {}).get("sizes", [0.33, 0.66])]
        deep_conf  = sb.get("deep", {"max_bb": 999, "sizes": [0.33, 0.66], "allow_pot": False})
        deep_sz    = [float(x) for x in deep_conf.get("sizes", [0.33, 0.66])]
        allow_pot  = bool(deep_conf.get("allow_pot", False))
    else:
        # Sensible defaults
        short_max, short_sz = 30.0, [0.25, 0.50]
        med_max,   med_sz   = 80.0, [0.33, 0.66]
        deep_sz,   allow_pot = [0.33, 0.66], False

    # Optional per-menu override to allow pot at deep
    mo = cfg.get("menu_overrides", {}).get(f"{group}.{role}", {})
    if "deep_allow_pot" in mo:
        allow_pot = bool(mo["deep_allow_pot"])

    # Apply buckets
    if s <= short_max:
        return sorted(set(short_sz))
    if s <= med_max:
        return [med_sz[0]] if not want_two else sorted(set(med_sz))
    # deep
    out = list(deep_sz)
    if allow_pot or want_has_pot:
        if 1.00 not in out:
            out.append(1.00)
    return sorted(set(out))

# --- PUBLIC API (unchanged signature/shape) ---
def build_contextual_bet_sizes(
    menu_id: Optional[str],
    *,
    stake: Stakes = Stakes.NL10,
    effective_stack_bb: Optional[float] = None,
) -> dict:
    """
    Build flop/turn/river bet-size structure for a given menu ID and stake.
    Stack-aware legalization uses STAKE_CFG when present; fallback to defaults.
    """
    key = (menu_id or "").strip()
    group, role = _parse_menu_id(key)

    cfg = STAKE_CFG.get(stake, STAKE_CFG[Stakes.NL10])
    RAISE_MULT = cfg["raise_mult"]
    ENABLE_ALLIN = cfg.get("flop_allin", True)

    desired = _desired_fracs_for(key, stake)
    sizes_frac: List[float] = _legalize_fracs_by_stack(
        desired,
        group=group,
        role=role,
        effective_stack_bb=effective_stack_bb,
        stake=stake,
    )

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

    # Limped multi: symmetric, no donk on OOP
    if group.startswith("limped_multi"):
        flop_cfg["oop"].pop("donk", None)
        flop_cfg["oop"].setdefault("bet", sizes_frac)

    TURN_RIVER_DISABLED = {"ip": {"bet": []}, "oop": {"bet": []}}
    return {"flop": flop_cfg, "turn": TURN_RIVER_DISABLED, "river": TURN_RIVER_DISABLED}
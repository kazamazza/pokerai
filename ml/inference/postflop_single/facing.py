from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from ml.inference.postflop_single.sizing import parse_bet_size_token

_POS_ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

def is_hero_ip(hero_pos: Optional[str], villain_pos: Optional[str]) -> bool:
    try:
        h = _POS_ORDER.index(str(hero_pos or "").upper())
        v = _POS_ORDER.index(str(villain_pos or "").upper())
        return h > v
    except Exception:
        return True

def street_name(street: int) -> str:
    return {0: "pre", 1: "flop", 2: "turn", 3: "river"}.get(int(street), "flop")

def derive_facing_from_history(actions_hist: Optional[List[str]], street: int, villain_pos: Optional[str]) -> Optional[bool]:
    if not actions_hist:
        return None
    sname = street_name(street)
    vpos = str(villain_pos or "").upper()
    for line in reversed(actions_hist):
        t = str(line or "").strip().lower()
        if not t.startswith(sname + ":"):
            continue
        is_villain = vpos.lower() in t
        aggr = ("bet" in t) or ("raise" in t) or ("donk" in t)
        if is_villain and aggr:
            return True
        if is_villain and (("check" in t) or ("call" in t) or ("fold" in t)):
            return False
        if (not is_villain) and (("bet" in t) or ("raise" in t)):
            return False
    return None

def infer_facing_and_size(req: Any, *, hero_is_ip: bool) -> tuple[bool, Optional[float], Dict[str, Any]]:
    """Returns (facing_bet, size_frac, debug)."""
    dbg: Dict[str, Any] = {}
    # 1) explicit fields
    frac = None
    fsf = getattr(req, "faced_size_frac", None) if not isinstance(req, dict) else req.get("faced_size_frac")
    fsp = getattr(req, "faced_size_pct", None) if not isinstance(req, dict) else req.get("faced_size_pct")
    if fsf is not None:
        try: frac = float(fsf)
        except Exception: pass
    elif fsp is not None:
        try: frac = float(fsp) / 100.0
        except Exception: pass
    fb = getattr(req, "facing_bet", None) if not isinstance(req, dict) else req.get("facing_bet")
    if isinstance(fb, bool):
        dbg["source"] = "explicit_flag"
        return True, frac, dbg
    # 2) history
    hist = getattr(req, "actions_hist", None) if not isinstance(req, dict) else req.get("actions_hist")
    street = int(getattr(req, "street", 1) if not isinstance(req, dict) else req.get("street", 1))
    villain_pos = getattr(req, "villain_pos", None) if not isinstance(req, dict) else req.get("villain_pos")
    f_hist = derive_facing_from_history(hist, street, villain_pos)
    if f_hist is True:
        # try parse size from last line
        try:
            last = str(hist[-1])
            frac_hist = parse_bet_size_token(last)
        except Exception:
            frac_hist = None
        dbg["source"] = "history"
        return True, (frac if frac is not None else frac_hist), dbg
    if f_hist is False:
        dbg["source"] = "history"
        return False, frac, dbg
    # 3) raw size hints
    raw = getattr(req, "raw", {}) or {} if not isinstance(req, dict) else req.get("raw", {}) or {}
    try:
        if frac is None:
            if "size_frac" in raw and raw["size_frac"] is not None:
                frac = float(raw["size_frac"])
            elif "size_pct" in raw and raw["size_pct"] is not None:
                frac = float(raw["size_pct"]) / 100.0
    except Exception:
        pass
    dbg["source"] = "default_root"
    return False, frac, dbg
POS_ORDER = {"UTG":1, "HJ":2, "CO":3, "BTN":4, "SB":0, "BB":-1}

def _canon_pos(p: str | None) -> str:
    if not p: return "BTN"
    p = str(p).strip().upper()
    return {
        "DEALER":"BTN","BUTTON":"BTN","UTG1":"HJ","UTG+1":"HJ","UTG2":"CO","UTG+2":"CO",
        "SMALL BLIND":"SB","BIG BLIND":"BB"
    }.get(p, p)

def _who_acts_first_preflop(a: str, b: str) -> str:
    # highest order acts first (BTN>CO>HJ>UTG>SB>BB)
    return a if POS_ORDER.get(a, -99) >= POS_ORDER.get(b, -99) else b

def deduce_preflop_context(
    hero_pos: str | None,
    villain_pos: str | None,
    actions_hist: list[str] | None,
    raw: dict | None,
    pot_bb: float | None,
) -> dict:
    """
    Returns:
      {
        "hero_pos": str, "villain_pos": str,
        "opener_pos": str | "",
        "opener_action": "RAISE" | "LIMP" | "NONE",
        "facing_open": bool
      }
    """
    hero = _canon_pos(hero_pos)
    vill = _canon_pos(villain_pos)
    first = _who_acts_first_preflop(hero, vill)  # likely BTN unless BvB

    # 1) explicit flags take precedence
    r = dict(raw or {})
    if isinstance(r.get("facing_open"), bool):
        facing_open = bool(r["facing_open"])
    else:
        # 2) derive from actions_hist if present
        ah = [str(a).upper() for a in (actions_hist or [])]
        if any(a.startswith("RAISE") for a in ah):
            facing_open = True
        elif any(a in ("LIMP","CALL","CHECK") for a in ah) and not any(a.startswith("RAISE") for a in ah):
            facing_open = False
        else:
            # 3) OPTIONAL pot-size heuristic (very rough; use only as last resort)
            # For 2.5bb open in SRP, total pot preflop ~ 3.5–4.0bb (rake/antes ignored).
            p = float(pot_bb or 0.0)
            facing_open = p >= 3.2  # tweak if you want
    # opener resolution
    if facing_open:
        opener_pos = first  # the one who acted first opened
        opener_action = "RAISE"
    else:
        # either no action yet or limp/check; if you still want to tag opener:
        opener_pos = first if any(a in ("LIMP","CALL","CHECK") for a in (actions_hist or [])) else ""
        opener_action = "LIMP" if opener_pos else "NONE"

    return {
        "hero_pos": hero,
        "villain_pos": vill,
        "opener_pos": opener_pos,
        "opener_action": opener_action,
        "facing_open": facing_open,
    }
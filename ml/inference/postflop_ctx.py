# helpers_postflop_ctx.py
from ml.inference.context_infer import ContextInferer

ALLOWED_PAIRS = {
    "VS_OPEN": {
        ("UTG","SB"),("UTG","BB"),
        ("HJ","SB"), ("HJ","BB"),
        ("CO","SB"), ("CO","BB"),
        ("BTN","SB"),("BTN","BB"),
    },
    "BLIND_VS_STEAL": {
        ("BTN","SB"),("BTN","BB"),
        ("CO","SB"), ("CO","BB"),
    },
    "VS_3BET": {
        ("BTN","SB"),("BTN","BB"),
        ("CO","SB"), ("CO","BB"),
    },
    "VS_4BET": {
        ("BTN","SB"),("BTN","BB"),
        ("CO","SB"), ("CO","BB"),
    },
    "LIMPED_SINGLE": {("BB","SB")},
}

def _flop_ip(ip: str, oop: str) -> tuple[str,str]:
    ip = (ip or "").upper(); oop = (oop or "").upper()
    if {ip,oop} == {"SB","BB"}: return ("BB","SB")
    order = ["SB","BB","UTG","HJ","CO","BTN"]
    try:
        return (ip,oop) if order.index(ip) > order.index(oop) else (oop,ip)
    except ValueError:
        return (ip,oop)

def _pair_in_domain(ip: str, oop: str) -> bool:
    for ctx, pairs in ALLOWED_PAIRS.items():
        if (ip, oop) in pairs:
            return True
    return False

def validate_pair_or_raise(req) -> tuple[str,str]:
    ip, oop = _flop_ip(req.hero_pos, req.villain_pos)
    if _pair_in_domain(ip, oop):
        return ip, oop
    allowed = sorted({p for pairs in ALLOWED_PAIRS.values() for p in pairs})
    raise ValueError(
        f"unseen postflop pair for training domain: ip={ip}, oop={oop}. "
        f"Allowed pairs: {allowed}. Adjust hero_pos/villain_pos or extend scenarios."
    )

def ensure_ctx_and_action_seq(req) -> None:
    street = int(getattr(req, "street", 1) or 1)
    if street == 0:
        setattr(req, "action_seq", getattr(req, "action_seq", []) or [])
        return

    ip, oop = validate_pair_or_raise(req)

    ctx = getattr(req, "ctx", None)
    if not ctx:
        ctx_infer, reason = ContextInferer.infer_with_reason(req)
        if not ctx_infer:
            # tolerant fallback by seat-pair when history/pot are unhelpful
            if (ip, oop) in ALLOWED_PAIRS["LIMPED_SINGLE"]:
                ctx_infer = "LIMPED_SINGLE"
            elif (ip, oop) in ALLOWED_PAIRS["VS_OPEN"] or (ip, oop) in ALLOWED_PAIRS.get("BLIND_VS_STEAL", set()):
                ctx_infer = "VS_OPEN"
            elif (ip, oop) in ALLOWED_PAIRS["VS_3BET"]:
                ctx_infer = "VS_3BET"
            elif (ip, oop) in ALLOWED_PAIRS["VS_4BET"]:
                ctx_infer = "VS_4BET"
            else:
                raise ValueError(f"ctx inference failed: {reason}")
        ctx = ctx_infer

    ctx = str(ctx).upper()
    if ctx == "BLIND_VS_STEAL":
        ctx = "VS_OPEN"

    setattr(req, "ctx", ctx)
    setattr(req, "action_seq", {
        "VS_OPEN":        ["RAISE","CALL",""],
        "VS_3BET":        ["RAISE","3BET","CALL"],
        "VS_4BET":        ["RAISE","3BET","4BET"],
        "LIMPED_SINGLE":  ["LIMP","CHECK",""],
    }.get(ctx, ["","",""]))
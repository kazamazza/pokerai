# --- helpers/postflop_ctx.py (or near PolicyRequest) ---
from ml.inference.policy.types import PolicyRequest


def infer_postflop_ctx(req: "PolicyRequest") -> str:
    """
    Infer preflop lineage/context for postflop decisions.
    Returns one of {"VS_OPEN","VS_3BET","VS_4BET","LIMPED_SINGLE","LIMPED_MULTI"}.
    Fallback is "VS_OPEN".
    """
    # 1) explicit override in raw (highest priority)
    raw_ctx = (req.raw or {}).get("ctx")
    if raw_ctx:
        ctx = str(raw_ctx).strip().upper()
        # normalize aliases
        if ctx in {"BLIND_VS_STEAL", "BVS", "STEAL"}:
            return "VS_OPEN"
        if ctx in {"VS_OPEN","VS_3BET","VS_4BET","LIMPED_SINGLE","LIMPED_MULTI"}:
            return ctx
        # unknown override → safe fallback
        return "VS_OPEN"

    # 2) derive from actions_hist (accept list/tuple/str/mixed)
    ah = (req.actions_hist
          or (req.raw or {}).get("actions_hist")
          or [])
    if isinstance(ah, str):
        # allow comma/space separated
        tokens = [t for t in re.split(r"[,\s]+", ah) if t]
    else:
        tokens = list(ah)

    toks = [str(x).strip().upper() for x in tokens if x is not None]

    # direct signals
    if any("4BET" in t for t in toks):
        return "VS_4BET"
    if any("3BET" in t for t in toks):
        return "VS_3BET"

    # limp logic
    limp_count   = sum(1 for t in toks if "LIMP" in t)
    raise_count  = sum(1 for t in toks if "RAISE" in t)
    call_count   = sum(1 for t in toks if t == "CALL" or "CALL " in t)
    overcalls    = max(0, call_count - int(raise_count > 0))  # rough “multiway-ish” indicator

    if limp_count >= 2:
        return "LIMPED_MULTI"
    if limp_count == 1:
        # one limp + no raise + (at least one overcall) → multi
        if raise_count == 0 and overcalls >= 1:
            return "LIMPED_MULTI"
        # one limp + raise (got called or not) → usually single-limp tree
        return "LIMPED_SINGLE"

    # treat blind-vs-steal or no special tokens as SRP
    return "VS_OPEN"
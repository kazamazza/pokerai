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
        return str(raw_ctx).strip().upper()

    # 2) derive from actions_hist (normalize tokens)
    ah = (req.actions_hist or (req.raw or {}).get("actions_hist") or [])
    toks = [str(x).strip().upper() for x in ah]

    # simple signals
    if any("4BET" in t for t in toks):
        return "VS_4BET"
    if any("3BET" in t for t in toks):
        return "VS_3BET"

    if any("LIMP" in t for t in toks):
        # crude split single vs multi: multiple limps or limp + overcalls → multi
        limp_count = sum(1 for t in toks if "LIMP" in t)
        overcalls  = sum(1 for t in toks if t == "CALL")
        if limp_count > 1 or (limp_count >= 1 and overcalls >= 1 and not any("RAISE" in t for t in toks)):
            return "LIMPED_MULTI"
    # if we saw exactly one limp and then a raise that got called, it’s typically SRP anyway
    # else default:
    return "VS_OPEN"
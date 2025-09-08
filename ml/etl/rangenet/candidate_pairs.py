ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
IDX = {p:i for i,p in enumerate(ORDER)}

def canon_pos(p: str) -> str:
    return str(p).upper()

def dist(opener: str, defender: str) -> int:
    """seats strictly between opener and defender (mod 6)"""
    return (IDX[defender] - IDX[opener] - 1) % len(ORDER)

# Simple, sensible opener-substitution ladders (don’t change defender)
OPEN_SUBS = {
    "SB":  ["CO", "HJ", "UTG"],  # SB opener fallback → earlier positions
    "BTN": ["CO", "HJ", "UTG"],
    "CO":  ["HJ", "UTG"],
    "HJ":  ["UTG"],
    # UTG has no earlier fallback
}

def candidate_pairs(ip: str, oop: str, *, ctx: str, allow_pair_subs: bool):
    """
    Yield (cand_ip, cand_oop, level_name, substituted:bool) in priority order.
    level_name is just for meta/debug.
    """
    ip = canon_pos(ip); oop = canon_pos(oop)
    ctx = str(ctx).upper()

    def ok_for_ctx(op, df):
        if ctx == "LIMP_MULTI":
            # require at least one seat between opener and defender
            return dist(op, df) >= 1
        # SRP / LIMP_SINGLE have no extra spacing constraint
        return True

    # 1) exact pair
    if ok_for_ctx(ip, oop):
        yield (ip, oop, "exact_pair", False)

    if not allow_pair_subs:
        return

    # 2) opener-substituted candidates (keep same defender)
    for sub in OPEN_SUBS.get(ip, []):
        if ok_for_ctx(sub, oop):
            yield (sub, oop, "sub_opener", True)
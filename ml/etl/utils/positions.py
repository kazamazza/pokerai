# --- positions helpers ---

VALID_SRP_PAIRS = {
    ("UTG","BB"),
    ("HJ","BB"),
    ("CO","BB"),
    ("BTN","BB"),
    ("SB","BB"),
}

# Keep these in sync with what you actually have in SPH
VALID_LIMP_SINGLE_PAIRS = {
    ("UTG","BB"), ("HJ","BB"), ("CO","BB"), ("BTN","BB"), ("SB","BB"),
    ("CO","SB"), ("BTN","SB"), ("UTG","SB"),
}

VALID_LIMP_MULTI_PAIRS = {
    ("UTG","BB"), ("HJ","BB"), ("CO","BB"), ("BTN","BB"), ("SB","BB"),
    ("CO","SB"),
}

def canon_pair(ip: str, oop: str) -> tuple[str, str]:
    """Uppercase + trims; leaves order intact (IP,OOP)."""
    return (str(ip).strip().upper(), str(oop).strip().upper())

def valid_pairs_for_ctx(ctx: str) -> set[tuple[str,str]]:
    ctx = str(ctx).upper()
    if ctx == "SRP":
        return VALID_SRP_PAIRS
    if ctx == "LIMP_SINGLE":
        return VALID_LIMP_SINGLE_PAIRS
    if ctx == "LIMP_MULTI":
        return VALID_LIMP_MULTI_PAIRS
    # default: empty (unknown ctx)
    return set()

def sanitize_position_pairs(pairs_in: list[tuple[str,str]], ctx: str) -> list[tuple[str,str]]:
    """Canonicalize and filter to those legal for this context; dedupe."""
    legal = valid_pairs_for_ctx(ctx)
    seen = set()
    out = []
    for ip, oop in pairs_in:
        ip2, oop2 = canon_pair(ip, oop)
        if ip2 == oop2:
            continue
        if (ip2, oop2) not in legal:
            # skip illegal for this context
            continue
        if (ip2, oop2) not in seen:
            seen.add((ip2, oop2))
            out.append((ip2, oop2))
    return out
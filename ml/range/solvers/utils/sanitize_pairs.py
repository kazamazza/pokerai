from typing import List, Set, Tuple, Sequence

Pair = Tuple[str, str]
POS_SET: Set[str] = {"UTG","HJ","CO","BTN","SB","BB"}

def canon_pair(ip: str, oop: str) -> Pair:
    return str(ip).strip().upper(), str(oop).strip().upper()

# Normalize common labels
CTX_ALIASES = {
    "SRP": "SRP", "OPEN": "SRP", "VS_OPEN": "SRP", "VS_OPEN_RFI": "SRP",
    "BLIND_VS_STEAL": "BLIND_VS_STEAL", "BVS": "BLIND_VS_STEAL",
    "VS_3BET": "VS_3BET", "3BET": "VS_3BET",
    "VS_4BET": "VS_4BET", "4BET": "VS_4BET",
    "LIMPED_SINGLE": "LIMP_SINGLE", "LIMP_SINGLE": "LIMP_SINGLE",
    "LIMPED_MULTI": "LIMP_MULTI",   "LIMP_MULTI": "LIMP_MULTI",
    "VS_CBET": "VS_CBET", "VS_CBET_TURN": "VS_CBET_TURN",
    "VS_CHECK_RAISE": "VS_CHECK_RAISE", "VS_DONK": "VS_DONK",
}

# Legal (IP,OOP) pairs per context (lean, safe defaults)
VALID_SRP_PAIRS: Set[Pair] = {
    ("UTG","BB"), ("HJ","BB"), ("CO","BB"), ("BTN","BB"),
    ("BTN","SB"),          # BTN open, SB flat (BB folded)
    ("BB","SB"),           # SB open, BB flat → IP=BB, OOP=SB
    ("UTG","SB"), ("HJ","SB"), ("CO","SB"),
}

VALID_3BET_PAIRS: Set[Pair] = {
    ("BTN","BB"), ("CO","BB"), ("HJ","BB"), ("UTG","BB"),
    ("BTN","SB"), ("CO","SB"), ("HJ","SB"),
    ("BB","BTN"), ("SB","BTN"), ("CO","BTN"),
}

VALID_4BET_PAIRS: Set[Pair] = {
    ("BTN","BB"), ("BB","BTN"),
    ("BTN","SB"), ("SB","BTN"),
}

VALID_BVS_PAIRS: Set[Pair] = {
    ("BTN","BB"), ("BTN","SB"), ("CO","BB"),
}

VALID_LIMP_SINGLE_PAIRS: Set[Pair] = {
    ("BB","SB"),           # SB limps, BB checks → IP=BB
}

VALID_LIMP_MULTI_PAIRS: Set[Pair] = {
    ("BB","SB"), ("BTN","SB"),
}

def valid_pairs_for_ctx(ctx: str) -> Set[Pair]:
    key = CTX_ALIASES.get(str(ctx).upper(), str(ctx).upper())
    if key == "SRP":           return VALID_SRP_PAIRS
    if key == "VS_3BET":       return VALID_3BET_PAIRS
    if key == "VS_4BET":       return VALID_4BET_PAIRS
    if key == "BLIND_VS_STEAL":return VALID_BVS_PAIRS
    if key == "LIMP_SINGLE":   return VALID_LIMP_SINGLE_PAIRS
    if key == "LIMP_MULTI":    return VALID_LIMP_MULTI_PAIRS
    # Unknown/unmodeled contexts → no legal pairs
    return set()

def sanitize_position_pairs(pairs_in: Sequence[Pair], ctx: str) -> List[Pair]:
    """Canonicalize, validate, and filter (IP,OOP) for a given ctx; dedupe."""
    legal = valid_pairs_for_ctx(ctx)
    seen: Set[Pair] = set()
    out: List[Pair] = []
    for ip, oop in pairs_in:
        ip2, oop2 = canon_pair(ip, oop)
        # basic seat validation + distinctness
        if ip2 == oop2 or ip2 not in POS_SET or oop2 not in POS_SET:
            continue
        # STRICT: if the legal set is empty or does not contain the pair, skip
        if (ip2, oop2) not in legal:
            continue
        if (ip2, oop2) not in seen:
            seen.add((ip2, oop2))
            out.append((ip2, oop2))
    return out
from typing import List, Set, Tuple, Sequence

from ml.etl.utils.positions import canon_pos

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
    "LIMPED_SINGLE": "LIMPED_SINGLE", "LIMPED_SINGLE": "LIMPED_SINGLE",
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


VALID_LIMP_SINGLE_PAIRS: Set[Pair] = {
    ("BB","SB"),           # SB limps, BB checks → IP=BB
}

VALID_BVS_PAIRS: Set[Pair] = {
    ("BTN","BB"), ("BTN","SB"), ("CO","BB"), ("CO","SB")  # add ("CO","SB") if you’ll use it
}

VALID_LIMP_MULTI_PAIRS: Set[Pair] = {
    ("BB","SB"), ("BTN","SB"), ("BTN","BB")  # add ("BTN","BB")
}

# --- New helper ---
def to_ip_oop_from_clockwise(ctx: str, a: str, b: str) -> tuple[str, str]:
    """
    Convert a human-friendly pair (a,b) to internal (IP,OOP) for the flop,
    based on context. Accepts SB/BB order for limped pots and opener/defender
    order for SRP/BvS/3bet/4bet.
    """
    A, B = canon_pos(a), canon_pos(b)
    c = str(ctx).upper()

    if c in ("LIMPED_SINGLE", "LIMPED_SINGLE"):
        # SB limps, BB checks → flop IP=BB, OOP=SB
        return ("BB", "SB")  # regardless of input order

    if c in ("LIMPED_MULTI", "LIMP_MULTI"):
        if (A, B) == ("BB", "SB"):   return ("BB", "SB")
        if (A, B) == ("BTN", "SB"):   return ("BTN", "SB")
        if (A, B) == ("BTN", "BB"):   return ("BTN", "BB")
        return (A, B)

    # SRP and relatives: treat input as (opener, defender) and derive IP/OOP:
    # Late pos vs blinds → opener (BTN/CO) is IP on flop; SB vs BB SRP_OOP etc. handled by pairs list you supply.
    late = {"BTN","CO"}; blinds = {"SB","BB"}
    opener, defender = A, B
    if opener in late and defender in blinds:
        return (opener, defender)  # opener IP, blind OOP
    # SB opened vs BB call (SRP_OOP): flop IP=BB, OOP=SB
    if opener == "SB" and defender == "BB":
        return ("BB", "SB")
    # BTN opened vs SB call: flop IP=BTN, OOP=SB
    if opener == "BTN" and defender == "SB":
        return ("BTN", "SB")
    # Defaults: assume first is IP
    return (A, B)

def valid_pairs_for_ctx(ctx: str) -> Set[Pair]:
    key = CTX_ALIASES.get(str(ctx).upper(), str(ctx).upper())
    if key == "SRP":           return VALID_SRP_PAIRS
    if key == "VS_3BET":       return VALID_3BET_PAIRS
    if key == "VS_4BET":       return VALID_4BET_PAIRS
    if key == "BLIND_VS_STEAL":return VALID_BVS_PAIRS
    if key == "LIMPED_SINGLE":   return VALID_LIMP_SINGLE_PAIRS
    if key == "LIMP_MULTI":    return VALID_LIMP_MULTI_PAIRS
    # Unknown/unmodeled contexts → no legal pairs
    return set()

def sanitize_position_pairs(pairs_in: Sequence[Pair], ctx: str) -> list[Pair]:
    legal = valid_pairs_for_ctx(ctx)     # still defined in INTERNAL (IP,OOP)
    out, seen = [], set()
    for a, b in pairs_in:
        ip2, oop2 = to_ip_oop_from_clockwise(ctx, a, b)  # <-- normalize here
        if ip2 == oop2 or ip2 not in POS_SET or oop2 not in POS_SET:
            continue
        if legal and (ip2, oop2) not in legal:
            continue
        if (ip2, oop2) not in seen:
            seen.add((ip2, oop2))
            out.append((ip2, oop2))
    return out
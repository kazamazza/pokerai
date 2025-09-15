from typing import Set, Tuple

Pair = Tuple[str, str]

# Normalize various spellings to canonical keys we handle
CTX_ALIASES = {
    "SRP": "SRP",
    "OPEN": "SRP",
    "VS_OPEN": "SRP",

    "VS_3BET": "VS_3BET",
    "3BET": "VS_3BET",

    "VS_4BET": "VS_4BET",
    "4BET": "VS_4BET",

    "BLIND_VS_STEAL": "BLIND_VS_STEAL",
    "BVS": "BLIND_VS_STEAL",

    "LIMPED_SINGLE": "LIMP_SINGLE",
    "LIMP_SINGLE": "LIMP_SINGLE",

    "LIMPED_MULTI": "LIMP_MULTI",
    "LIMP_MULTI": "LIMP_MULTI",

    "VS_CBET": "VS_CBET",
    "VS_CBET_TURN": "VS_CBET_TURN",
    "VS_CHECK_RAISE": "VS_CHECK_RAISE",
    "VS_DONK": "VS_DONK",
}

def canon_pair(ip: str, oop: str) -> Pair:
    return (str(ip).strip().upper(), str(oop).strip().upper())

# --- Minimal legal (IP,OOP) sets per context (flop positions) ---

# SRP heads-up: opener vs blind; include SB-open vs BB-call (IP=BB)
VALID_SRP_PAIRS: Set[Pair] = {
    ("BTN", "BB"), ("CO", "BB"), ("HJ", "BB"), ("UTG", "BB"),
    ("SB",  "BB"),  # SB opens, BB calls → OOP=BB? (preflop), but on flop IP is BB if SB checked
    ("BB",  "SB"),  # SB opens, BB calls → IP=BB on flop; include this explicit ordering
}

# Typical 3-bet pots (caller can be IP vs BB/SB 3-bettor, or caller OOP vs IP 3-bettor)
VALID_3BET_PAIRS: Set[Pair] = {
    # Caller IP vs OOP 3-bettor in blinds
    ("BTN", "BB"), ("CO", "BB"), ("HJ", "BB"), ("UTG", "BB"),
    ("BTN", "SB"), ("CO", "SB"),
    # Caller OOP vs IP 3-bettor (IP 3-bets BTN vs blinds or BTN vs CO)
    ("BB", "BTN"), ("SB", "BTN"), ("CO", "BTN"),
}

# Lean 4-bet coverage (both orientations)
VALID_4BET_PAIRS: Set[Pair] = {
    ("BTN", "BB"), ("BB", "BTN"),
    ("BTN", "SB"), ("SB", "BTN"),
}

# Blind vs steal (SRP subclass): stealer IP vs blind OOP
VALID_BVS_PAIRS: Set[Pair] = {
    ("BTN", "BB"), ("BTN", "SB"), ("CO", "BB"),
}

# Limped pots (keep conservative)
VALID_LIMP_SINGLE_PAIRS: Set[Pair] = {
    ("BB", "SB"),  # SB limps, BB checks → IP=BB on flop
}

VALID_LIMP_MULTI_PAIRS: Set[Pair] = {
    # Minimal heads-up extraction from multiway (optional)
    ("BB", "SB"), ("BTN", "SB"),
}


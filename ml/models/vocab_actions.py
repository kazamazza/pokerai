from typing import List, Optional, Sequence, Set

# -------------------------
# Postflop (from sidecars)
# -------------------------
ROOT_ACTION_VOCAB: List[str] = [
    "CHECK",
    "BET_25",
    "BET_33",
    "BET_50",
    "BET_66",
    "BET_75",
    "BET_100",
    "DONK_33",
]

FACING_ACTION_VOCAB: List[str] = [
    "FOLD",
    "CALL",
    "RAISE_150",
    "RAISE_200",
    "RAISE_300",
    "RAISE_400",
    "RAISE_500",
    "ALLIN",
]

# -------------------------
# Preflop (centi-bb schema)
# -------------------------

PREFLOP_ACTION_VOCAB = [
    "FOLD", "CHECK", "CALL", "ALLIN",
    "OPEN_200", "OPEN_250", "OPEN_300",
    "RAISE_600", "RAISE_750", "RAISE_900", "RAISE_1200",
]

# -------------------------
# Helpers (preflop tokens)
# -------------------------
def encode_open_cbb(size_bb: float) -> str:
    return f"OPEN_{int(round(float(size_bb) * 100))}"

def encode_raise_total_cbb(total_bb: float) -> str:
    return f"RAISE_{int(round(float(total_bb) * 100))}"

def decode_open_cbb(tok: str) -> Optional[float]:
    if not tok.startswith("OPEN_"): return None
    try: return int(tok.split("_", 1)[1]) / 100.0
    except Exception: return None

def decode_raise_total_cbb(tok: str) -> Optional[float]:
    if not tok.startswith("RAISE_"): return None
    try: return int(tok.split("_", 1)[1]) / 100.0
    except Exception: return None

def is_postflop_root(tok: str) -> bool:
    return tok in set(ROOT_ACTION_VOCAB)

def is_postflop_facing(tok: str) -> bool:
    return tok in set(FACING_ACTION_VOCAB)

def is_preflop(tok: str) -> bool:
    return tok in set(PREFLOP_ACTION_VOCAB)
from typing import List, Optional

# ============================================================
# V1 FINAL ACTION VOCABS
# Sizing standard:
#   pot_pre := pot size before the actor's action
#   size_frac := (incremental chips put in by actor) / pot_pre
#   faced_frac := (villain incremental bet) / pot_pre_villain_bet
#   raise_to_frac := (actor's total bet size after raise) / pot_pre
# ============================================================

# -------------------------
# Postflop (root = no bet faced)
# -------------------------
# Must match your bet menus: 0.25, 0.33, 0.50, 0.66, 0.75, 1.25
ROOT_ACTION_VOCAB: List[str] = [
    "CHECK",
    "BET_25",
    "BET_33",
    "BET_50",
    "BET_66",
    "BET_75",
    "BET_125",
]

# -------------------------
# Postflop (facing = bet/raise faced)
# -------------------------
# Minimal v1 to reduce sparse classes; maps cleanly to raise_mult [2.0, 3.0, 4.5]
FACING_ACTION_VOCAB: List[str] = [
    "FOLD",
    "CALL",
    "RAISE_TO_200",
    "RAISE_TO_300",
    "RAISE_TO_450",
    "ALLIN",
]

# -------------------------
# Preflop (centi-bb schema)
# -------------------------
PREFLOP_ACTION_VOCAB: List[str] = [
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
    if not tok.startswith("OPEN_"):
        return None
    try:
        return int(tok.split("_", 1)[1]) / 100.0
    except Exception:
        return None

def decode_raise_total_cbb(tok: str) -> Optional[float]:
    if not tok.startswith("RAISE_"):
        return None
    try:
        return int(tok.split("_", 1)[1]) / 100.0
    except Exception:
        return None

# -------------------------
# Helpers (membership)
# -------------------------
_ROOT_SET = set(ROOT_ACTION_VOCAB)
_FACING_SET = set(FACING_ACTION_VOCAB)
_PREFLOP_SET = set(PREFLOP_ACTION_VOCAB)

def is_postflop_root(tok: str) -> bool:
    return tok in _ROOT_SET

def is_postflop_facing(tok: str) -> bool:
    return tok in _FACING_SET

def is_preflop(tok: str) -> bool:
    return tok in _PREFLOP_SET
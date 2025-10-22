ROOT_ACTION_VOCAB = [
    "CHECK",     # allowed when no bet faced
    "BET_25",
    "BET_33",
    "BET_50",
    "BET_66",
    "BET_75",
    "BET_100",
    "DONK_33",   # OOP-only root action
]

# --- Facing actions (for responding to a bet) ---
FACING_ACTION_VOCAB = [
    "FOLD",
    "CALL",
    "RAISE_150",
    "RAISE_200",
    "RAISE_300",
    "RAISE_400",
    "RAISE_500",
    "ALLIN",
]
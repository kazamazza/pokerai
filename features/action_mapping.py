RAW_ACTION_MAP = {
    # Fold variations
    "folds": "fold",
    "folded": "fold",
    "folded before flop": "fold",
    "folded before turn": "fold",
    "folded before river": "fold",

    # Call variations
    "calls": "call",
    "called": "call",

    # Raise / bet
    "raises": "raise",
    "raised": "raise",
    "bets": "raise",  # You can later split this if needed
}

ACTIONS = [
    "fold", "call", "raise",  # Core normalized actions
    "open", "3bet", "4bet", "jam", "iso", "limp"  # For future inference/annotation
]

def normalize_action(raw: str) -> tuple[str | None, float | None]:
    """Normalize a raw action line to a canonical form and extract amount."""
    raw = raw.lower()
    for phrase, normalized in RAW_ACTION_MAP.items():
        if phrase in raw:
            try:
                amount = float(raw.split("$")[-1])
            except Exception:
                amount = None
            return normalized, amount
    return None, None
def normalize_action(raw: str) -> tuple[str | None, float | None]:
    raw = raw.lower().strip()
    RAW_ACTION_MAP = {
        "folds": "fold",
        "folded": "fold",
        "folded before flop": "fold",
        "folded before turn": "fold",
        "folded before river": "fold",
        "calls": "call",
        "called": "call",
        "raises": "raise",
        "raised": "raise",
        "bets": "raise",
    }
    for phrase, normalized in RAW_ACTION_MAP.items():
        if phrase in raw:
            amounts = [float(s.strip("$")) for s in raw.split() if "$" in s]
            return normalized, amounts[-1] if amounts else None
    return None, None
# preflop/matchups.py


# (IP_position, OOP_position)
MATCHUPS = [
    # Standard open vs defense
    ("MP", "UTG"),
    ("CO", "UTG"), ("CO", "MP"),
    ("BTN", "UTG"), ("BTN", "MP"), ("BTN", "CO"),
    ("SB", "UTG"), ("SB", "MP"), ("SB", "CO"), ("SB", "BTN"),
    ("BB", "UTG"), ("BB", "MP"), ("BB", "CO"), ("BB", "BTN"),

    # Blind vs blind
    ("BB", "SB"),

    # Optional: SB vs open (for shove/fold sims or vs ISO)
    ("SB", "BB"),
]
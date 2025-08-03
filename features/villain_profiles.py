VILLAIN_PROFILES = [
    "GTO",
    "TAG",
    "LAG",
    "NIT",
    "MANIAC",
    "FISH"
]

# Ranges are representative, based on community heuristics
VILLAIN_PROFILE_STATS = {
    "GTO": {
        "vpip": (0.20, 0.28),
        "pfr": (0.18, 0.25),
        "three_bet": (0.05, 0.10),
        "flop_cbet": (0.65, 0.75),
        "fold_to_cbet": (0.40, 0.55),
        "wtsd": (0.22, 0.28),
        "wsd": (0.48, 0.55)
    },
    "TAG": {
        "vpip": (0.18, 0.25),
        "pfr": (0.16, 0.22),
        "three_bet": (0.05, 0.09),
        "flop_cbet": (0.65, 0.80),
        "fold_to_cbet": (0.45, 0.60),
        "wtsd": (0.25, 0.30),
        "wsd": (0.50, 0.58)
    },
    "LAG": {
        "vpip": (0.28, 0.40),
        "pfr": (0.24, 0.35),
        "three_bet": (0.08, 0.14),
        "flop_cbet": (0.70, 0.90),
        "fold_to_cbet": (0.35, 0.50),
        "wtsd": (0.30, 0.40),
        "wsd": (0.45, 0.55)
    },
    "NIT": {
        "vpip": (0.10, 0.18),
        "pfr": (0.08, 0.15),
        "three_bet": (0.02, 0.05),
        "flop_cbet": (0.50, 0.65),
        "fold_to_cbet": (0.55, 0.70),
        "wtsd": (0.15, 0.22),
        "wsd": (0.50, 0.60)
    },
    "MANIAC": {
        "vpip": (0.45, 0.60),
        "pfr": (0.35, 0.50),
        "three_bet": (0.12, 0.20),
        "flop_cbet": (0.80, 0.95),
        "fold_to_cbet": (0.20, 0.40),
        "wtsd": (0.35, 0.50),
        "wsd": (0.30, 0.45)
    },
    "FISH": {
        "vpip": (0.35, 0.50),
        "pfr": (0.10, 0.20),
        "three_bet": (0.02, 0.06),
        "flop_cbet": (0.40, 0.60),
        "fold_to_cbet": (0.60, 0.80),
        "wtsd": (0.40, 0.55),
        "wsd": (0.30, 0.45)
    }
}
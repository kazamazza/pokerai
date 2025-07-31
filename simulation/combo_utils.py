def get_hero_combo_string(hole_cards: list) -> str:
    """
    Convert two hole cards like ['Td', '9d'] into a combo string: 'T9s', 'AKo', etc.
    Assumes cards are strings like 'Td', 'Ah', etc.
    """
    if len(hole_cards) != 2:
        raise ValueError("Expected exactly 2 hole cards.")

    r1, s1 = hole_cards[0][0], hole_cards[0][1]
    r2, s2 = hole_cards[1][0], hole_cards[1][1]

    # Normalize order: higher rank first
    if "AKQJT98765432".index(r1) < "AKQJT98765432".index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1

    if r1 == r2:
        return r1 + r2  # e.g., "TT"
    elif s1 == s2:
        return r1 + r2 + "s"
    else:
        return r1 + r2 + "o"
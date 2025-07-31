def expand_range_syntax(range_str: str) -> list[str]:
    """
    Expand shorthand poker combo syntax into a list of concrete hand strings.
    Handles eval7.HandRange entries of the form ((Card, Card), weight)

    Returns:
        List[str]: like ['AhKd', '2d2c', ...]
    """
    import eval7
    hr = eval7.HandRange(range_str)
    combos = []

    for entry in hr:
        try:
            # Entry is ((Card, Card), weight)
            cards, weight = entry
            if not isinstance(cards, tuple) or len(cards) != 2:
                raise ValueError("Combo did not contain exactly 2 cards.")
            c1, c2 = str(cards[0]), str(cards[1])
            hand = c1 + c2
            if len(hand) == 4:
                combos.append(hand)
            else:
                raise ValueError(f"Bad hand format: {hand}")
        except Exception as e:
            print(f"[SKIP] Malformed combo: {entry} → {e}")
            continue

    assert all(len(c) == 4 for c in combos), "❌ Some combos are not in format 'AhKd'"
    return combos
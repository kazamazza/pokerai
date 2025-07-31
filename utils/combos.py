from typing import List

def get_169_combo_list() -> List[str]:
    """
    Returns the canonical list of 169 starting hand combos in standard order:
    - 13 pairs
    - 78 suited hands
    - 78 offsuit hands
    """
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    combos = []

    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                # Pocket pairs
                combos.append(f"{r1}{r2}")
            elif i < j:
                # Suited hands
                combos.append(f"{r1}{r2}s")
            else:
                # Offsuit hands
                combos.append(f"{r1}{r2}o")

    return combos
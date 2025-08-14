import random
from typing import List

import eval7

def compute_hand_vs_range_equity(hand_combo: str,
                                 opponent_combos: List[str],
                                 iterations: int = 5000) -> float:
    """
    Monte Carlo equity of a shorthand hand (e.g. 'AKs') vs a list of concrete opp combos (['AsKd', ...]).
    Returns equity in [0,1].
    """
    hero_concrete = expand_combo_string(hand_combo)
    if not hero_concrete or not opponent_combos:
        return 0.0

    hero_wins = 0.0
    trials = 0

    for _ in range(iterations):
        # Pick random hero and opponent concrete combos; reject if they overlap
        for _guard in range(20):  # avoid infinite loops in rare overlap streaks
            hstr = random.choice(hero_concrete)
            ostr = random.choice(opponent_combos)
            try:
                h1, h2 = eval7.Card(hstr[:2]), eval7.Card(hstr[2:])
                o1, o2 = eval7.Card(ostr[:2]), eval7.Card(ostr[2:])
            except Exception:
                continue
            if len({h1, h2, o1, o2}) == 4:
                break
        else:
            continue  # failed to find non-overlapping in 20 tries; skip this iter

        # Build and shuffle deck, remove used cards, draw random board
        deck = eval7.Deck()
        used = {h1, h2, o1, o2}
        deck.cards = [c for c in deck if c not in used]
        random.shuffle(deck.cards)            # <-- crucial
        board = deck.draw(5)                  # random 5-card board

        hv = eval7.evaluate([h1, h2] + board)
        ov = eval7.evaluate([o1, o2] + board)

        if hv > ov:
            hero_wins += 1.0
        elif hv == ov:
            hero_wins += 0.5
        trials += 1

    return hero_wins / trials if trials else 0.0


def expand_combo_string(combo: str) -> list[str]:
    """
    Expand shorthand into unique concrete combos.
    - 'AKs' -> 4 combos
    - 'AKo' -> 12 combos
    - 'AK'  -> 16 combos
    - '22'  -> 6 combos
    """
    suits = 'shdc'
    out = []

    if len(combo) == 3:
        r1, r2, suitedness = combo[0], combo[1], combo[2]
    elif len(combo) == 2:
        r1, r2 = combo[0], combo[1]
        suitedness = None
    else:
        return []

    # Pairs
    if r1 == r2:
        for i, s1 in enumerate(suits):
            for s2 in suits[i+1:]:
                out.append(f"{r1}{s1}{r2}{s2}")
        return out  # 6 combos

    # Non-pairs
    for i, s1 in enumerate(suits):
        for j, s2 in enumerate(suits):
            if s1 == s2 and suitedness == 'o':
                continue            # offsuit only
            if s1 != s2 and suitedness == 's':
                continue            # suited only

            # enforce unique ordering: for offsuit or unspecified, keep s1 < s2
            if suitedness in (None, 'o') and s1 >= s2:
                continue
            # for suited, we already require s1 == s2

            out.append(f"{r1}{s1}{r2}{s2}")
    return out
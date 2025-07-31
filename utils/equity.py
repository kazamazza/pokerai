import random
from typing import List

import eval7


def compute_hand_vs_range_equity(hand_combo: str, opponent_combos: List[str], iterations: int = 5000) -> float:
    """
    Estimate equity of a single hand (e.g., 'AKs') vs a list of opponent concrete combos (e.g. ['AhKd', '2h2c', ...]).

    Args:
        hand_combo (str): Combo like 'AKs', 'JTo', etc.
        opponent_combos (List[str]): List of concrete hands like 'AhKd', '2h2c', ...
        iterations (int): Number of Monte Carlo simulations

    Returns:
        float: Estimated equity [0.0, 1.0]
    """
    hero_combos = expand_combo_string(hand_combo)

    if not hero_combos or not opponent_combos:
        print(f"[ERROR] No valid combos → hero: {hand_combo}, opp: {len(opponent_combos)}")
        return 0.0

    hero_wins = 0
    total = 0

    for _ in range(iterations):
        deck = eval7.Deck()

        random.shuffle(hero_combos)
        random.shuffle(opponent_combos)

        found_valid = False

        for hero_str in hero_combos:
            for opp_str in opponent_combos:
                try:
                    hero = [eval7.Card(hero_str[:2]), eval7.Card(hero_str[2:])]
                    opp  = [eval7.Card(opp_str[:2]), eval7.Card(opp_str[2:])]
                except Exception:
                    continue

                if set(hero) & set(opp):
                    continue  # overlapping cards

                used = hero + opp
                deck.cards = [card for card in deck if card not in used]
                board = deck.peek(5)

                hero_val = eval7.evaluate(hero + board)
                opp_val  = eval7.evaluate(opp + board)

                if hero_val > opp_val:
                    hero_wins += 1
                elif hero_val == opp_val:
                    hero_wins += 0.5

                total += 1
                found_valid = True
                break

            if found_valid:
                break

    if total == 0:
        print(f"[WARNING] No valid simulations ran for {hand_combo} vs opponent.")
        return 0.0

    return hero_wins / total


def expand_combo_string(combo: str) -> list[str]:
    """
    Expands shorthand combos into all valid concrete hand combinations.

    Examples:
        'AKs' -> ['AsKs', 'AhKh', 'AdKd', 'AcKc']
        'AKo' -> ['AsKd', 'AsKh', ..., 'AcKh']
        'AK'  -> all 16 combos (suited + offsuit)
        '22'  -> all 6 pairs: ['2s2h', '2s2d', '2s2c', ...]
    """
    suits = 'shdc'
    cards = []

    # Extract rank(s) and suitedness
    if len(combo) == 3:
        r1, r2, suitedness = combo[0], combo[1], combo[2]
    elif len(combo) == 2:
        r1, r2 = combo[0], combo[1]
        suitedness = None
    else:
        return []  # invalid combo string

    for s1 in suits:
        for s2 in suits:
            if r1 == r2:
                if s1 >= s2:
                    continue  # avoid duplicates in pairs
            elif s1 == s2 and suitedness == 'o':
                continue  # skip suited when offsuit requested
            elif s1 != s2 and suitedness == 's':
                continue  # skip offsuit when suited requested

            c1, c2 = r1 + s1, r2 + s2
            if c1 == c2:
                continue  # skip identical cards
            cards.append(c1 + c2)

    return cards
# ml/utils/cards.py
from typing import List
import random
import eval7

from ml.utils.constants import RANKS, SUITS

DECK_ORDER = [r+s for r in RANKS for s in SUITS]
CARD_INDEX = {c:i for i,c in enumerate(DECK_ORDER)}

# Build the exact deck order used during cluster build:
_DECK_ORDER = [r+s for r in RANKS for s in SUITS]
_IDX = {c:i for i,c in enumerate(_DECK_ORDER)}

def deck_index(card_str: str) -> int:
    """Return the stable deck-order index used by cluster builder."""
    return _IDX[card_str]

def card_str(c: "eval7.Card") -> str:
    return str(c)  # e.g., "As"

def hand_string_to_cards(hand: str) -> List["eval7.Card"]:
    """
    Convert 'AKs'/'AQo'/'TT' into a concrete 2-card combo consistent with suitedness.
    """
    if len(hand) == 3 and hand.endswith(("s","o")):
        r1, r2, tag = hand[0], hand[1], hand[2]
        suited = (tag == "s")
    else:
        # pair like 'TT'
        r1, r2, suited = hand[0], hand[1], False

    suits = list(SUITS)
    if r1 == r2:
        s1, s2 = random.sample(suits, 2)
        return [eval7.Card(r1 + s1), eval7.Card(r2 + s2)]
    if suited:
        s = random.choice(suits)
        return [eval7.Card(r1 + s), eval7.Card(r2 + s)]
    else:
        s1, s2 = random.sample(suits, 2)
        return [eval7.Card(r1 + s1), eval7.Card(r2 + s2)]
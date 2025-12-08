# file: ml/etl/ev/lib/sampling.py
from __future__ import annotations
import random
from typing import Iterable, List, Set
import eval7

def _norm_card_list(exclude: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for c in exclude or []:
        s = str(c).strip()
        if len(s) == 2:
            out.add(s[0].upper() + s[1].lower())
    return out

def _deck_str() -> List[str]:
    # eval7.Deck() -> iterable of Cards; convert to ['Ah','Kd',...]
    return [str(c) for c in list(eval7.Deck())]

def sample_random_hand_excluding(*, exclude: Iterable[str] = (), rng: random.Random | None = None) -> str:
    """
    Returns a 4-char hero hand like 'AhKd', excluding any 2-char codes in `exclude` (e.g. 'Ah','Kd').
    """
    rng = rng or random
    ex = _norm_card_list(exclude)
    deck = [c for c in _deck_str() if c not in ex]
    # sample 2 distinct cards
    i = rng.randrange(len(deck))
    c1 = deck.pop(i)
    j = rng.randrange(len(deck))
    c2 = deck[j]
    return c1 + c2  # 'AhKd'

def sample_random_flop_excluding(*, exclude: Iterable[str] = (), rng: random.Random | None = None) -> str:
    """
    Returns a 6-char flop string like '2d3d4s', excluding any 2-char codes in `exclude`.
    """
    rng = rng or random
    ex = _norm_card_list(exclude)
    deck = [c for c in _deck_str() if c not in ex]
    # sample 3 distinct cards
    i = rng.randrange(len(deck))
    c1 = deck.pop(i)
    j = rng.randrange(len(deck))
    c2 = deck.pop(j)
    k = rng.randrange(len(deck))
    c3 = deck[k]
    return c1 + c2 + c3
import random
from typing import List, Tuple, Dict
import eval7


def mc_equity_vs_range(hero_hand_id: int,
                       flop_samples: List[Tuple[str,str,str]],
                       opp_combo_weights: Dict[str, float],
                       trials_per_flop: int = 200) -> float:
    from ml.etl.equity.hands import canon_hand_from_id
    hero = canon_hand_from_id(hero_hand_id)  # 'AKs' etc.
    # Pick a random suited instantiation for hero (cheap and unbiased over many flops)
    hero_combos = expand_canon_to_combos(hero)
    total_w = 0.0
    win_tie = 0.0

    for f in flop_samples:
        board3 = [eval7.Card(c) for c in f]
        for _ in range(trials_per_flop):
            hc = eval7.Card(random.choice(hero_combos[:2]))  # slight speed hack: pick one combo each draw
            hd = eval7.Card(random.choice(hero_combos[2:]))
            # sample an opponent combo proportionally to weight, avoiding dead cards
            opp = sample_opp_combo(opp_combo_weights, dead={str(hc),str(hd),*f})
            if not opp: continue
            oc, od = eval7.Card(opp[:2]), eval7.Card(opp[2:])

            deck = [c for c in all_deck_cards() if str(c) not in {str(hc),str(hd),opp[:2],opp[2:],*f}]
            random.shuffle(deck)
            turn, river = deck[0], deck[1]

            hero_hand = [hc, hd]
            opp_hand  = [oc, od]
            board = board3 + [turn, river]
            hv = eval7.evaluate(hero_hand + board)
            ov = eval7.evaluate(opp_hand  + board)
            w = 1.0  # (you can use opp_combo_weights[opp] as a weight per draw too)
            total_w += w
            if hv > ov: win_tie += w
            elif hv == ov: win_tie += w*0.5
    return win_tie / max(1e-9, total_w)

# helpers you'd implement once:
def expand_canon_to_combos(canon: str) -> List[str]: ...
def sample_opp_combo(weights: Dict[str,float], dead: set[str]) -> str | None: ...
def all_deck_cards(): return [eval7.Card(r+s) for r in "23456789TJQKA" for s in "cdhs"]
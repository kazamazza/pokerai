from collections import Counter
from typing import List
from flop.features.schema import FlopFeatures, StraightDrawPotential, Connectivity, TextureClass


def extract_flop_features(board: List[str]) -> FlopFeatures:
    """
    Given a 3-card flop, extract key strategic features for clustering or strategy abstraction.
    """
    assert len(board) == 3, "Board must be a flop (3 cards)"
    ranks = [card[0] for card in board]
    suits = [card[1] for card in board]

    rank_order = '23456789TJQKA'
    rank_values = [rank_order.index(r) for r in ranks]

    # Paired board?
    rank_counts = Counter(ranks)
    is_paired = any(v >= 2 for v in rank_counts.values())

    # Suited board?
    suit_counts = Counter(suits)
    is_monotone = max(suit_counts.values()) == 3
    is_two_tone = max(suit_counts.values()) == 2

    # Flush draw?
    has_flush_draw = is_two_tone

    # High cards?
    high_ranks = {'T', 'J', 'Q', 'K', 'A'}
    high_card_count = sum(r in high_ranks for r in ranks)
    has_ace = 'A' in ranks
    has_king = 'K' in ranks

    # Straight draw logic
    sorted_vals = sorted(rank_values)
    gaps = sorted_vals[2] - sorted_vals[0]

    if gaps == 2 and sorted_vals[1] - sorted_vals[0] == 1:
        straight_draw = StraightDrawPotential.OPEN_ENDED
    elif gaps <= 4:
        straight_draw = StraightDrawPotential.GUTSHOT
    else:
        straight_draw = StraightDrawPotential.NONE

    # Connectivity (coarse score)
    if gaps <= 3:
        connectivity = Connectivity.HIGH
    elif gaps <= 5:
        connectivity = Connectivity.MEDIUM
    else:
        connectivity = Connectivity.LOW

    # Wetness score (simple version)
    wetness_score = 0.0
    wetness_score += 0.2 if is_monotone else 0.1 if is_two_tone else 0.0
    wetness_score += 0.15 if is_paired else 0.0
    wetness_score += 0.15 if straight_draw != StraightDrawPotential.NONE else 0.0
    wetness_score += 0.1 * high_card_count / 3

    # Texture class assignment (simplified rule-based)
    if is_monotone:
        texture_class = TextureClass.TRIPLE_SUITED
    elif is_paired:
        texture_class = TextureClass.PAIR_HIGH if any(r in ['A', 'K', 'Q'] for r, count in rank_counts.items() if count >= 2) else TextureClass.PAIR_LOW
    elif high_card_count == 3:
        texture_class = TextureClass.ACE_HIGH
    elif connectivity == Connectivity.HIGH and high_card_count <= 1:
        texture_class = TextureClass.LOW_CONNECT
    elif wetness_score >= 0.6:
        texture_class = TextureClass.WET
    elif wetness_score <= 0.3:
        texture_class = TextureClass.DRY
    else:
        texture_class = TextureClass.SEMI_DRY

    return FlopFeatures(
        is_paired=is_paired,
        has_ace=has_ace,
        has_king=has_king,
        has_flush_draw=has_flush_draw,
        is_monotone=is_monotone,
        is_two_tone=is_two_tone,
        high_card_count=high_card_count,
        straight_draw_potential=straight_draw,
        wetness_score=wetness_score,
        connectivity=connectivity,
        texture_class=texture_class
    )
from collections import Counter
from typing import List

from features.types import RANK_ORDER

class BoardNormalizer:

    def normalize(self, board: List[str]) -> str:
        ranks = sorted([card[:-1] for card in board], key=lambda r: RANK_ORDER[r])
        suits = [card[-1] for card in board]

        # Suit texture
        suit_set = set(suits)
        if len(suit_set) == 1:
            suit_texture = "monotone"
        elif len(suit_set) == 2:
            suit_texture = "twotone"
        else:
            suit_texture = "rainbow"

        # Structure
        rank_counts = Counter(ranks)
        values = sorted([RANK_ORDER[r] for r in ranks])
        is_triplet = 3 in rank_counts.values()
        is_paired = 2 in rank_counts.values()
        is_connected = values[2] - values[0] <= 4
        is_high = all(v >= 10 for v in values)
        is_low = all(v <= 5 for v in values)

        if is_triplet:
            structure = "triplet"
        elif is_paired:
            structure = "paired"
        elif is_connected:
            structure = "connected"
        elif is_high:
            structure = "high"
        elif is_low:
            structure = "low"
        else:
            structure = "uncoordinated"

        cluster_key = f"{'-'.join(ranks)}:{structure}:{suit_texture}"
        return cluster_key
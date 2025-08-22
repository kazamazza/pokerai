from dataclasses import dataclass
from typing import List, Dict

from ml.config.types_hands import SUITS
from ml.features.boards.board_parsing import parse_board


@dataclass
class BoardFeatures:
    n_cards: int              # 3/4/5
    paired: int               # 1 if any rank repeats
    trips: int                # 1 if any rank occurs 3 times
    quads: int                # 1 if any rank occurs 4 times
    max_rank: int             # 0..12
    second_max_rank: int      # 0..12 or -1
    low_straight_potential: int  # rough proxy
    high_straight_potential: int
    n_suits: int              # number of distinct suits on board (1..4)
    has_2suited: int          # at least two of same suit
    has_3suited: int          # at least three of same suit
    monotone: int             # 1 if all same suit
    connectivity: float       # avg rank gap (lower is more connected)
    wheel_present: int        # A-2-3 / A-2-3-4 patterns possible

    def to_vector(self) -> List[float]:
        return [
            float(self.n_cards),
            float(self.paired), float(self.trips), float(self.quads),
            float(self.max_rank), float(self.second_max_rank),
            float(self.low_straight_potential), float(self.high_straight_potential),
            float(self.n_suits), float(self.has_2suited), float(self.has_3suited),
            float(self.monotone), float(self.connectivity), float(self.wheel_present),
        ]


def featurize_board(board_str: str) -> BoardFeatures:
    b = parse_board(board_str)
    n = len(b)
    ranks = sorted([r for (r, _) in b])
    suits = [s for (_, s) in b]

    # Rank multiplicities
    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    multiplicities = sorted(counts.values(), reverse=True)
    paired = 1 if any(c == 2 for c in multiplicities) else 0
    trips  = 1 if any(c == 3 for c in multiplicities) else 0
    quads  = 1 if any(c == 4 for c in multiplicities) else 0

    max_rank = ranks[-1]
    second_max = ranks[-2] if n >= 2 else -1

    # Suits
    distinct_suits = len(set(suits))
    has_2suited = 1 if any(suits.count(s) >= 2 for s in SUITS) else 0
    has_3suited = 1 if any(suits.count(s) >= 3 for s in SUITS) else 0
    monotone = 1 if distinct_suits == 1 else 0

    # Connectivity: mean abs diff of consecutive ranks
    if n > 1:
        gaps = [ranks[i+1] - ranks[i] for i in range(n-1)]
        connectivity = float(sum(gaps)) / len(gaps)
    else:
        connectivity = 0.0

    # Straight-ish heuristics (very rough)
    low_straight_potential = 1 if ranks[0] <= 3 else 0       # presence of very low ranks
    high_straight_potential = 1 if max_rank >= 10 else 0     # presence of high ranks

    # Wheel (A present + low cards)
    wheel_present = 1 if (12 in ranks and any(r in ranks for r in [0,1,2,3])) else 0

    return BoardFeatures(
        n_cards=n,
        paired=paired, trips=trips, quads=quads,
        max_rank=max_rank, second_max_rank=second_max,
        low_straight_potential=low_straight_potential,
        high_straight_potential=high_straight_potential,
        n_suits=distinct_suits, has_2suited=has_2suited, has_3suited=has_3suited,
        monotone=monotone, connectivity=connectivity, wheel_present=wheel_present,
    )
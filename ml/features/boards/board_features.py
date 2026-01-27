from dataclasses import dataclass
from typing import List, Dict, Tuple

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

    # Straight potential (now meaningful, still simple + deterministic)
    low_straight_potential: int   # 1 if some 5-rank window in A-low space contains >=3 ranks
    high_straight_potential: int  # 1 if some 5-rank window (normal) contains >=3 ranks

    n_suits: int              # number of distinct suits on board (1..4)
    has_2suited: int          # at least two of same suit
    has_3suited: int          # at least three of same suit
    monotone: int             # 1 if all same suit

    connectivity: float       # avg gap between consecutive UNIQUE ranks (lower = more connected)
    wheel_present: int        # A + (2/3/4/5) presence (A234/ A2345 type texture)

    def to_vector(self) -> List[float]:
        return [
            float(self.n_cards),
            float(self.paired), float(self.trips), float(self.quads),
            float(self.max_rank), float(self.second_max_rank),
            float(self.low_straight_potential), float(self.high_straight_potential),
            float(self.n_suits), float(self.has_2suited), float(self.has_3suited),
            float(self.monotone), float(self.connectivity), float(self.wheel_present),
        ]


def _rank_multiplicity_flags(ranks: List[int]) -> Tuple[int, int, int]:
    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    multiplicities = sorted(counts.values(), reverse=True)
    paired = 1 if any(c == 2 for c in multiplicities) else 0
    trips  = 1 if any(c == 3 for c in multiplicities) else 0
    quads  = 1 if any(c == 4 for c in multiplicities) else 0
    return paired, trips, quads


def _straight_window_hit(unique_ranks: List[int], *, use_wheel_ace_low: bool) -> int:
    """
    Returns 1 if there exists a 5-rank window that contains >=3 of the board ranks.
    - If use_wheel_ace_low=True, treat Ace (12) also as rank -1 (i.e. below 2) for wheel texture.
    """
    if not unique_ranks:
        return 0

    ranks_set = set(unique_ranks)

    # For wheel-aware evaluation, add ace as -1 proxy so windows like A2345 are detectable.
    # We don't mutate the real ranks; only this window scan sees the proxy.
    if use_wheel_ace_low and 12 in ranks_set:
        ranks_set = set(ranks_set)
        ranks_set.add(-1)

    # Scan windows of length 5 across plausible start points.
    # Normal ranks are 0..12; with ace_low proxy, min can be -1.
    lo = min(ranks_set)
    hi = max(ranks_set)

    # We only need to scan a bounded range; extend slightly so -1..12 works.
    start_min = min(-1, lo)
    start_max = max(8, hi)  # 8 is last start for window [8..12] in normal space

    for start in range(start_min, start_max + 1):
        window = {start, start + 1, start + 2, start + 3, start + 4}
        hit = len(window & ranks_set)
        if hit >= 3:
            return 1
    return 0


def _unique_rank_connectivity(unique_ranks_sorted: List[int]) -> float:
    """
    Mean gap between consecutive UNIQUE ranks.
    Paired boards should not artificially look 'connected' or 'disconnected' due to duplicates.
    """
    if len(unique_ranks_sorted) <= 1:
        return 0.0
    gaps = [unique_ranks_sorted[i + 1] - unique_ranks_sorted[i] for i in range(len(unique_ranks_sorted) - 1)]
    return float(sum(gaps)) / float(len(gaps))


def featurize_board(board_str: str) -> BoardFeatures:
    b = parse_board(board_str)
    n = len(b)
    if n not in (3, 4, 5):
        raise ValueError(f"Expected 3/4/5 cards board; got n_cards={n} from {board_str!r}")

    ranks = [r for (r, _) in b]
    suits = [s for (_, s) in b]

    ranks_sorted = sorted(ranks)
    unique_ranks_sorted = sorted(set(ranks_sorted))

    paired, trips, quads = _rank_multiplicity_flags(ranks_sorted)

    max_rank = ranks_sorted[-1]
    second_max = ranks_sorted[-2] if n >= 2 else -1

    # Suit counts (one pass)
    suit_counts: Dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    distinct_suits = len(suit_counts)
    has_2suited = 1 if any(cnt >= 2 for cnt in suit_counts.values()) else 0
    has_3suited = 1 if any(cnt >= 3 for cnt in suit_counts.values()) else 0
    monotone = 1 if distinct_suits == 1 else 0

    connectivity = _unique_rank_connectivity(unique_ranks_sorted)

    # Straight potential (>=3 ranks in some 5-card window)
    # "low" = wheel-aware (Ace can play as 1)
    # "high" = normal (Ace stays high)
    low_straight_potential = _straight_window_hit(unique_ranks_sorted, use_wheel_ace_low=True)
    high_straight_potential = _straight_window_hit(unique_ranks_sorted, use_wheel_ace_low=False)

    # Wheel presence: Ace + at least two low ranks among {2,3,4,5} (0..3)
    ranks_set = set(unique_ranks_sorted)
    wheel_low = len(ranks_set & {0, 1, 2, 3})
    wheel_present = 1 if (12 in ranks_set and wheel_low >= 2) else 0

    return BoardFeatures(
        n_cards=n,
        paired=paired,
        trips=trips,
        quads=quads,
        max_rank=max_rank,
        second_max_rank=second_max,
        low_straight_potential=low_straight_potential,
        high_straight_potential=high_straight_potential,
        n_suits=distinct_suits,
        has_2suited=has_2suited,
        has_3suited=has_3suited,
        monotone=monotone,
        connectivity=connectivity,
        wheel_present=wheel_present,
    )
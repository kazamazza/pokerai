from typing import Tuple, List

from ml.config.types_hands import RANK_TO_I, RANKS


def parse_card(card: str) -> Tuple[int, str]:
    """'Ac' -> (12, 'c')  (A=12, K=11, ..., 2=0)"""
    return (RANK_TO_I[card[0]], card[1])

def parse_board(s: str) -> List[Tuple[int, str]]:
    """
    Accept 'AcKd7d' (len 6), 'AcKd7d2h' (8), 'AcKd7d2h3c' (10).
    Returns list of (rank_index, suit).
    """
    if not s or len(s) % 2 != 0:
        raise ValueError(f"Bad board string: {s!r}")
    out = []
    for i in range(0, len(s), 2):
        out.append(parse_card(s[i:i+2]))
    return out

def board_to_str(board: List[Tuple[int, str]]) -> str:
    return "".join(RANKS[r] + s for (r, s) in board)
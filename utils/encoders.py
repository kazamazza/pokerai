from typing import List

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def one_hot_encode(value: str, categories: List[str]) -> List[float]:
    return [1.0 if value == cat else 0.0 for cat in categories]


def encode_card(card_str: str) -> int:
    """
    Encodes a single card string (e.g., "Ah", "2c") into an integer from 0–51.
    """
    rank = card_str[0]
    suit = card_str[1]
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def encode_board(board: List[str], max_size: int = 5) -> List[int]:
    """
    Encodes a board of up to 5 cards into a list of integers (padded with -1 if needed).
    """
    encoded = [encode_card(card) for card in board]
    while len(encoded) < max_size:
        encoded.append(-1)  # Pad with -1 for missing board cards (e.g., flop or turn only)
    return encoded

def encode_position(pos: str) -> int:
    mapping = {"UTG": 0, "MP": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}
    return mapping.get(pos.upper(), -1)

def encode_category(value: str, categories: List[str]) -> int:
    return categories.index(value.upper()) if value.upper() in categories else -1

# Helper: Encode label
def encode_label(label: str) -> int:
    mapping = {
        "USE_GTO": 0,
        "EXPLOIT_BLUFF": 1,
        "EXPLOIT_FOLD": 2,
        "EXPLOIT_THIN_VALUE": 3
    }
    return mapping[label]
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

def make_board_mask_52(board: str) -> list[float]:
    """52-dim binary mask of cards present on the board."""
    mask = [0.0] * 52
    s = str(board or "").strip()
    if len(s) % 2 != 0:
        return mask
    for i in range(0, len(s), 2):
        r, u = s[i].upper(), s[i + 1].lower()
        try:
            ri = _RANKS.index(r)
            si = _SUITS.index(u)
            idx = ri * 4 + si  # rank-major order (13 ranks × 4 suits)
            mask[idx] = 1.0
        except ValueError:
            continue
    return mask
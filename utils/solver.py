def parse_board_string(board_compact: str) -> list[str]:
    """
    '4c8dQh' -> ['4c','8d','Qh']
    """
    if not board_compact:
        return []
    if "," in board_compact:  # already comma-separated
        return [t.strip() for t in board_compact.split(",") if t.strip()]
    out = []
    s = board_compact.strip()
    for i in range(0, len(s), 2):
        out.append(s[i:i+2])
    return out


def combos_to_range_str(combos: list[str]) -> str:
    """
    ['AA','AKs','A5o',...] -> 'AA,AKs,A5o,...'
    """
    return ",".join(combos)
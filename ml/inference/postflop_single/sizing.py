from __future__ import annotations
import re
from typing import Optional, Sequence

KNOWN_SIZE_FRACS = (0.25, 0.33, 0.50, 0.66, 0.75, 1.00)

def parse_bet_size_token(tok: str) -> Optional[float]:
    """Accept 'BET 33', 'BET_33', 'DONK 0.5', 'BET 100%', returns fraction or None."""
    if not tok:
        return None
    s = tok.strip().upper().replace("%", "").replace("_", " ")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)$", s)
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.5:
        val = val / 100.0
    # snap
    best = min(KNOWN_SIZE_FRACS, key=lambda c: abs(c - val))
    return best if abs(best - val) <= 0.05 else None

def nearest_size_bucket(frac: float, menu: Sequence[float] | None = None) -> float:
    """Snap to menu if provided, else to KNOWN_SIZE_FRACS."""
    base = menu if menu else KNOWN_SIZE_FRACS
    return min(base, key=lambda c: abs(float(c) - float(frac)))
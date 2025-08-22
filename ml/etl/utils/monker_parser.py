from pathlib import Path
from typing import Dict
from ml.config.types_hands import HAND_TO_ID


def parse_monker_range_text(text: str) -> Dict[str, float]:
    """
    Parse a Monker-ish line like:
      'AA:1.0,A2s:0.0,A2o:0.174,...'
    into { 'AA':1.0, 'A2s':0.0, 'A2o':0.174, ... }
    Ignores unknown tokens and normalizes if needed.
    """
    d: Dict[str, float] = {}
    text = text.strip()
    if not text:
        return d
    parts = text.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        try:
            val = float(v.strip())
        except ValueError:
            continue
        if k in HAND_TO_ID:
            d[k] = val

    # Normalize if the sum is > 0 and not ~1
    s = sum(d.values())
    if s > 0 and abs(s - 1.0) > 1e-6:
        scale = 1.0 / s
        for k in d:
            d[k] *= scale
    return d

def load_range_file(path: Path) -> Dict[str, float]:
    """
    Monker exports often have a single line of 'hand:prob,hand:prob,...'.
    If multiple lines, we merge (last wins) or average — here we take LAST non-empty.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # pick the last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    return parse_monker_range_text(lines[-1])
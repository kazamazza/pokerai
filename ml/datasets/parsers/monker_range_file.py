# ml/datasets/parsers/monker_range_file.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

from ml.core.types import RANKS, SUITS


def _is_token(tok: str) -> bool:
    # Valid tokens: "AA", "AKs", "AKo", "A5s", etc.
    if len(tok) == 2 and tok[0] in RANKS and tok[1] in RANKS:
        return True
    if len(tok) == 3 and tok[0] in RANKS and tok[1] in RANKS and tok[2] in ("s", "o"):
        return True
    return False

def parse_monker_line(line: str) -> Dict[str, float]:
    """
    Parse a single Monker 'token:weight' CSV line into a 169-grid dict.
    Keeps the *last* occurrence if duplicates appear.
    """
    by_token: Dict[str, float] = {}
    line = line.strip()
    if not line:
        return by_token
    # handle possible trailing commas or spaces
    parts = [p for p in line.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        tok_raw, w_raw = part.split(":", 1)
        tok = tok_raw.strip()
        if not _is_token(tok):
            continue
        try:
            w = float(w_raw.strip())
        except ValueError:
            continue
        # clip to [0,1] just in case
        if w < 0.0: w = 0.0
        if w > 1.0: w = 1.0
        by_token[tok] = w
    return by_token

def read_monker_file(path: str | Path) -> Dict[str, float]:
    """
    Read the file and parse the first non-empty line as the range.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Some vendors put key=value headers; skip those.
        if ":" in line and any(c.isalpha() for c in line):
            return parse_monker_line(line)
    return {}

def expand_169_to_1326(by_token: Dict[str, float]) -> Dict[str, float]:
    """
    Expand tokens like 'AKs' to all suited combos (4), 'AKo' to offsuit combos (12),
    pairs 'AA' to 6 combos. Key format: 'AsKs', 'AcKd', etc.
    Each combo inherits the token's weight.
    """
    out: Dict[str, float] = {}
    for tok, w in by_token.items():
        r1, r2 = tok[0], tok[1]
        if len(tok) == 2:  # pair, 6 combos
            for i, s1 in enumerate(SUITS):
                for j, s2 in enumerate(SUITS):
                    if i >= j:
                        continue  # avoid same card / unordered pairs
                    out[f"{r1}{s1}{r2}{s2}"] = w  # e.g. AsAh, AcAd, ...
        else:
            suited = (tok[2] == "s")
            offsuit = (tok[2] == "o")
            if suited:
                for s in SUITS:
                    out[f"{r1}{s}{r2}{s}"] = w  # e.g. AsKs, AcKc, ...
            elif offsuit:
                for s1 in SUITS:
                    for s2 in SUITS:
                        if s1 == s2:
                            continue
                        out[f"{r1}{s1}{r2}{s2}"] = w  # e.g. AsKd, AhKc, ...
            else:
                # Shouldn't happen, but if we ever see a mixed token, skip.
                pass
    return out
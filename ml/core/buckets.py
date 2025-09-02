# ml/exploit/buckets.py
from typing import Sequence, Optional

SPR_BINS_DEFAULT  = [0, 2, 4, 7, 100]
RATE_BINS_DEFAULT = [0.0, 0.2, 0.4, 0.7, 1.01]

# We use the last index as UNKNOWN bucket, i.e. len(bins)-1
UNKNOWN_BIN_SENTINEL = 9  # optional legacy; prefer dynamic unknown at len(bins)-1

def bucketize_spr(v: float, spr_bins: Sequence[float] = SPR_BINS_DEFAULT) -> int:
    # returns 0..(len-2) for real bins
    for i in range(len(spr_bins) - 1):
        if spr_bins[i] <= v < spr_bins[i + 1]:
            return i
    return len(spr_bins) - 2  # clamp to last real bin

def bucketize_rate(p: Optional[float], rate_bins: Sequence[float] = RATE_BINS_DEFAULT) -> int:
    if p is None or not (0.0 <= p <= 1.0):
        # unknown → put in a dedicated unknown bucket at the *end*
        return len(rate_bins) - 1
    for i in range(len(rate_bins) - 1):
        if rate_bins[i] <= p < rate_bins[i + 1]:
            return i
    return len(rate_bins) - 2

def n_rate_bins(rate_bins: Sequence[float] = RATE_BINS_DEFAULT) -> int:
    # real bins = len-1, unknown bin = +1 at the end
    return len(rate_bins)
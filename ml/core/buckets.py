from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Iterable
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinSpec:
    """Closed-open bins [e0,e1), [e1,e2), ... with the **last bin right-inclusive** [e_{n-1}, e_n].
    Unknown bucket is optional and, if used, is ALWAYS the last index.
    """
    edges: Tuple[float, ...]          # strictly increasing, length >= 2
    has_unknown: bool = False         # reserve final index for unknowns/invalids

    def validate(self) -> None:
        if len(self.edges) < 2:
            raise ValueError("BinSpec.edges must have ≥2 edges")
        if any(not math.isfinite(x) for x in self.edges):
            raise ValueError("BinSpec.edges must be finite")
        if any(self.edges[i] >= self.edges[i+1] for i in range(len(self.edges)-1)):
            raise ValueError("BinSpec.edges must be strictly increasing")

    @property
    def n_real_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def n_total_bins(self) -> int:
        return self.n_real_bins + (1 if self.has_unknown else 0)


def bucketize_scalar(
    v: Optional[float],
    spec: BinSpec,
) -> int:
    """Map a scalar to a bin index. NaN/None/invalid → unknown (if enabled) else clamp."""
    spec.validate()

    # Unknown handling
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return spec.n_real_bins if spec.has_unknown else 0  # unknown → last index if enabled

    # Last edge is right-inclusive
    if v >= spec.edges[-1]:
        return spec.n_real_bins - 1
    if v < spec.edges[0]:
        return 0

    # Binary search
    edges = spec.edges
    lo, hi = 0, len(edges) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if v < edges[mid+1]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bucketize_array(
    arr: Iterable[Optional[float]],
    spec: BinSpec,
) -> np.ndarray:
    """Vectorized bucketing for sequences/Series/ndarrays; returns int array."""
    spec.validate()
    a = np.asarray(list(arr), dtype=np.float64)
    out = np.empty(a.shape, dtype=np.int64)

    # unknowns
    unknown_idx = None
    if spec.has_unknown:
        unknown_idx = spec.n_real_bins

    mask_valid = np.isfinite(a)
    if spec.has_unknown:
        out[~mask_valid] = unknown_idx
    else:
        out[~mask_valid] = 0

    # valid values
    v = a[mask_valid]
    # clamp below first edge → 0
    lo_mask = v < spec.edges[0]
    # >= last edge → last real bin
    hi_mask = v >= spec.edges[-1]
    mid_mask = ~(lo_mask | hi_mask)

    out_valid = np.empty(v.shape, dtype=np.int64)
    out_valid[lo_mask] = 0
    out_valid[hi_mask] = spec.n_real_bins - 1

    # mid: use searchsorted on interior edges (right-open)
    if np.any(mid_mask):
        v_mid = v[mid_mask]
        # find i s.t. edges[i] <= v < edges[i+1]
        idx = np.searchsorted(spec.edges[1:-1], v_mid, side="right")
        out_valid[mid_mask] = idx

    out[mask_valid] = out_valid
    return out


# -------- Data-driven helpers (optional) --------

def quantile_edges(
    series: pd.Series,
    q: Sequence[float],
    min_width: float = 1e-9,
) -> Tuple[float, ...]:
    """Return strictly increasing edges from quantiles; deduplicate and widen ties."""
    qs = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    vals = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if vals.empty:
        # fallback
        return (0.0, 1.0)
    ed = np.quantile(vals, qs)
    # enforce strict increase
    edges = [float(ed[0])]
    for x in ed[1:]:
        if x <= edges[-1]:
            x = edges[-1] + min_width
        edges.append(float(x))
    return tuple(edges)


def propose_rate_bins(series: pd.Series) -> BinSpec:
    """Reasonable default for public HH rate features concentrated in a tight band."""
    # Example: split around median and upper tail; add unknown bucket
    edges = quantile_edges(series, q=[0.0, 0.5, 0.85, 1.0])
    return BinSpec(edges=edges, has_unknown=True)


def propose_spr_bins(series: pd.Series) -> BinSpec:
    """Handle degenerate SPR distributions (e.g., {0,100})."""
    uniq = pd.Series(series.unique())
    if set(np.round(uniq.fillna(-1).values, 3)) == {0.0, 100.0}:
        # explicit two bins for 0 and 100
        return BinSpec(edges=(0.0, 0.5, 100.0), has_unknown=False)
    # else generic: coarse low/med/high
    edges = quantile_edges(series, q=[0.0, 0.33, 0.66, 1.0])
    return BinSpec(edges=edges, has_unknown=False)
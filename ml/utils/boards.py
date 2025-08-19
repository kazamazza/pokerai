from __future__ import annotations
import json, itertools
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from ml.utils.constants import R2I, RANKS, SUITS

# Deterministic total order over 52 cards (e.g., As < Ah < Ad < Ac < Ks < ...)
CARD_ORDER: Dict[str,int] = {
    r + s: i for i, (r, s) in enumerate(itertools.product(RANKS, SUITS))
}

CardLike = Union[str, "eval7.Card"]

def _as_str(c: CardLike) -> str:
    return c if isinstance(c, str) else str(c)

def canon_card_key(c: str) -> str:
    """Normalize a 2-char card string like 'As', 'ah' -> 'As' (rank upper, suit lower)."""
    assert len(c) == 2
    return c[0].upper() + c[1].lower()

def canon_flop_key(cards3: List[str]) -> str:
    """
    Turn 3 card strings into a canonical key exactly matching how the cluster file was built.
    Our cluster builder enumerates deck in rank-major, suit-minor and takes combinations
    in ascending index; that is equivalent to **sorting by (rank_index, suit_index)**.
    """
    cs = [canon_card_key(c) for c in cards3]
    def idx(c):
        return (R2I[c[0]], SUITS.index(c[1]))
    cs.sort(key=idx)
    return "".join(cs)  # 'AhKdQc'

def _chunks3(key: str) -> Tuple[str,str,str]:
    assert len(key) == 6, f"expected 6-char key, got '{key}'"
    return key[0:2], key[2:4], key[4:6]

def expand_flop_permutations(key: str) -> List[str]:
    """
    Return all 6 permutations of the three 2-char chunks for robust lookup.
    """
    a,b,c = _chunks3(key)
    return ["".join(p) for p in itertools.permutations((a,b,c), 3)]

def load_cluster_map(path: Union[str, Path], expand_perms: bool = True) -> Tuple[Dict[str,int], Dict]:
    obj = json.loads(Path(path).read_text())
    base: Dict[str,int] = obj["clusters"]
    if expand_perms:
        expanded: Dict[str,int] = {}
        for k, cid in base.items():
            for alt in expand_flop_permutations(k):
                expanded[alt] = cid
        return expanded, obj["meta"]
    return base, obj["meta"]

def flop_cluster_id(cards3: List[CardLike], clusters: Dict[str,int]) -> int:
    """
    Robust lookup: try canonical key first; if not found, rely on expanded map.
    """
    key = canon_flop_key(cards3)
    return clusters.get(key, -1)

def _rank_idxs(cards: List[str]):  # ["Ah","Kd","Qc"] -> [0,1,2] (A,K,Q indices)
    return sorted(R2I[c[0]] for c in cards)

def _suit_hist(cards):             # [n♠, n♥, n♦, n♣]
    h = [0,0,0,0]
    for c in cards: h[SUITS.index(c[1])] += 1
    return h

def flop_features(board3: List[str]) -> List[float]:
    """Lite, 18-dim texture vector; deterministic and fast."""
    assert len(board3) == 3 and all(len(c)==2 for c in board3)
    r = _rank_idxs(board3)
    s = _suit_hist(board3)

    is_pair  = int(r[0]==r[1] or r[1]==r[2] or r[0]==r[2])
    is_trips = int(r[0]==r[1]==r[2])

    gaps = [r[i+1]-r[i] for i in range(2)]
    min_gap, max_gap = min(gaps), max(gaps)
    sum_gap = sum(gaps)

    broadway = sum(1 for x in r if x <= 4)  # A,K,Q,J,T
    wheel    = int(any(idx in r for idx in (0,8,9,10,11,12)))

    s3 = int(3 in s); s2 = int(2 in s); s1 = int(s.count(1) == 3)
    hi3 = sum(1 for x in r if x <= 4)
    lo3 = sum(1 for x in r if x >= 8)

    spread  = r[-1]-r[0]
    midness = abs(r[1] - (r[0]+r[-1])/2.0)

    N = 12.0
    f = [
        is_pair, is_trips,
        min_gap/ N, max_gap/ N, sum_gap/(2*N),
        broadway/3.0, wheel,
        s3, s2, s1,
        hi3/3.0, lo3/3.0,
        spread/ N, midness/(N/2),
        s[0]/3.0, s[1]/3.0, s[2]/3.0, s[3]/3.0,
    ]
    return f  # 18 dims

# ---------- Load / lookup ----------

def load_cluster_map(path: str) -> Tuple[Dict[str,int], Dict]:
    """Return (clusters_map, meta). clusters_map keys are canonical flop keys."""
    obj = json.loads(Path(path).read_text())
    clusters = obj["clusters"]
    meta     = obj["meta"]
    return clusters, meta

def nearest_centroid_idx(x: np.ndarray, centroids: np.ndarray) -> int:
    """Return index of nearest centroid (squared Euclidean)."""
    # centroids: [K, D], x: [D]
    d = centroids - x  # [K, D]
    dist2 = np.einsum("kd,kd->k", d, d)
    return int(np.argmin(dist2))

def flop_cluster_id_safe(cards3: List[str],
                         clusters: Dict[str,int],
                         meta: Dict) -> int:
    """
    Robust: try key lookup; if missing (e.g., different canonicalization),
    compute features and assign to nearest centroid.
    """
    key = canon_flop_key(cards3)
    cid = clusters.get(key, None)
    if cid is not None:
        return int(cid)

    # Fallback: nearest centroid on features
    cents = np.asarray(meta.get("centroids"), dtype=np.float32)
    x = np.asarray(flop_features(cards3), dtype=np.float32)
    return nearest_centroid_idx(x, cents)
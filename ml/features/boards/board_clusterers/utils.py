# Put this in something like: ml/board/clustering/utils.py
from __future__ import annotations
import random
from typing import Dict, List, Tuple

from ml.config.types_hands import RANKS, SUITS


def _all_52() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]

def _random_flops(rng: random.Random, n: int) -> List[List[str]]:
    deck = _all_52()
    out = []
    for _ in range(n):
        out.append(rng.sample(deck, 3))
    return out

def _flop_features(board: List[str]) -> List[float]:
    """
    Lightweight flop embedding for diversity picking:
      - rank counts (13)
      - suit counts (4)
      - texture flags: monotone, two_tone, paired (3)
      - straight-ish feature: number of distinct rank gaps across 3 ranks (1)
    Total: 21 dims
    """
    # rank + suit counts
    r_counts = [0]*13
    s_counts = [0]*4
    rank_to_i = {r:i for i,r in enumerate(RANKS)}
    suit_to_i = {s:i for i,s in enumerate(SUITS)}

    ranks = []
    suits = []
    for c in board:
        r, s = c[0], c[1]
        r_counts[rank_to_i[r]] += 1
        s_counts[suit_to_i[s]] += 1
        ranks.append(rank_to_i[r])  # 0..12 (A..2)
        suits.append(s)

    # texture flags
    mono = 1.0 if len(set(suits)) == 1 else 0.0
    two_tone = 1.0 if len(set(suits)) == 2 else 0.0
    paired = 1.0 if len(set(ranks)) < 3 else 0.0

    # straight-ish: unique gaps between sorted rank indices (after sorting ascending)
    rs = sorted(ranks)
    gaps = set([rs[1]-rs[0], rs[2]-rs[1]])
    straightish = float(len(gaps))  # 1 if equal gaps (like 5-6-7), 2 otherwise

    return [float(x) for x in (r_counts + s_counts + [mono, two_tone, paired, straightish])]

def _sqdist(a: List[float], b: List[float]) -> float:
    return sum((x - y)*(x - y) for x, y in zip(a, b))

def _pick_greedy_spread(items: List[Tuple[List[str], List[float]]], k: int, rng: random.Random) -> List[List[str]]:
    """
    K-center greedy: pick 1 random seed, then repeatedly add the board that
    has the largest min-distance to current chosen set.
    `items` = list of (board, feature_vector)
    """
    if not items:
        return []
    k = max(1, min(k, len(items)))

    # seed
    chosen_idx = [rng.randrange(len(items))]
    chosen = [items[chosen_idx[0]]]

    # precompute distances to speed up? (small enough; do it on the fly)
    while len(chosen) < k:
        # for each candidate, compute distance to nearest chosen
        best_idx = None
        best_score = -1.0
        for i, (_, feat) in enumerate(items):
            if i in chosen_idx:
                continue
            # min distance to any chosen
            d = min(_sqdist(feat, cf) for _, cf in chosen)
            if d > best_score:
                best_score = d
                best_idx = i
        chosen_idx.append(best_idx)
        chosen.append(items[best_idx])

    return [b for (b, _) in chosen]

def discover_representative_flops(
    clusterer,
    n_clusters_limit: int,
    boards_per_cluster: int,
    seed: int = 42,
    sample_pool: int = 20000,
) -> Dict[int, List[List[str]]]:
    """
    Sample `sample_pool` random flops, cluster them, then choose `boards_per_cluster`
    diverse representatives per cluster. Limit to top-`n_clusters_limit` by frequency.
    Returns: {cluster_id: [ [c1,c2,c3], ... ]}
    """
    rng = random.Random(seed)
    pool = _random_flops(rng, sample_pool)

    # Clusterer likely expects compact strings like "AhKd7c"
    compact = ["".join(b) for b in pool]
    labels = clusterer.predict(compact)  # must return list[int]-like

    # Count cluster frequencies
    freq: Dict[int, int] = {}
    for lab in labels:
        freq[lab] = freq.get(lab, 0) + 1
    # Select top clusters
    cluster_ids = [cid for cid, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]
    if n_clusters_limit > 0:
        cluster_ids = cluster_ids[:n_clusters_limit]

    # Group boards and compute features
    by_cluster: Dict[int, List[Tuple[List[str], List[float]]]] = {}
    for b, lab in zip(pool, labels):
        if lab not in cluster_ids:
            continue
        by_cluster.setdefault(lab, []).append((b, _flop_features(b)))

    # Pick representatives per cluster
    out: Dict[int, List[List[str]]] = {}
    for cid, items in by_cluster.items():
        reps = _pick_greedy_spread(items, boards_per_cluster, rng)
        out[cid] = reps
    return out
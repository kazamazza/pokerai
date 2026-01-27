from __future__ import annotations

import random
from typing import Dict, List, Iterable

from ml.config.types_hands import RANKS, SUITS


def _deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def _sample_board(rng: random.Random, k: int) -> str:
    cards = rng.sample(_deck(), k)
    return "".join(cards)  # canonical format


def _fill_representatives(
    *,
    clusterer,                 # expects .predict(List[str]) -> List[int]
    k: int,                    # 3/4/5
    n_clusters: int,
    boards_per_cluster: int,
    seed: int,
    max_samples: int,
) -> Dict[int, List[str]]:
    """
    Fill {cluster_id: [boards...]} trying to hit boards_per_cluster for every cluster.
    Deterministic RNG, strict board formatting.
    Raises if we cannot reach adequate coverage within max_samples.
    """
    rng = random.Random(int(seed))
    target = int(boards_per_cluster)
    if target <= 0:
        raise ValueError("boards_per_cluster must be >= 1")

    buckets: Dict[int, List[str]] = {i: [] for i in range(int(n_clusters))}
    filled_clusters = 0

    def is_full(cid: int) -> bool:
        return len(buckets[cid]) >= target

    for _ in range(int(max_samples)):
        if filled_clusters == n_clusters:
            break

        board = _sample_board(rng, k)
        cid = int(clusterer.predict([board])[0])  # strict, no fallback
        if cid < 0 or cid >= n_clusters:
            raise ValueError(f"Clusterer returned out-of-range cluster id {cid} (n_clusters={n_clusters})")

        if board in buckets[cid]:
            continue

        before = len(buckets[cid])
        if before < target:
            buckets[cid].append(board)
            after = len(buckets[cid])
            if before < target and after == target:
                filled_clusters += 1

    # sanity: ensure every cluster has at least 1, ideally target
    missing = [cid for cid, bs in buckets.items() if len(bs) < target]
    if missing:
        # You can choose to allow partials; for v1 I recommend FAIL FAST.
        raise RuntimeError(
            f"Could not fill representatives for {len(missing)}/{n_clusters} clusters "
            f"within max_samples={max_samples}. Example missing: {missing[:10]}"
        )

    return buckets


def discover_representative_flops(*, clusterer, n_clusters_limit: int, boards_per_cluster: int, seed: int,
                                 sample_pool: int = 50000) -> Dict[int, List[str]]:
    return _fill_representatives(
        clusterer=clusterer, k=3,
        n_clusters=int(n_clusters_limit),
        boards_per_cluster=int(boards_per_cluster),
        seed=int(seed),
        max_samples=int(sample_pool),
    )

def discover_representative_turns(*, clusterer, n_clusters_limit: int, boards_per_cluster: int, seed: int,
                                 sample_pool: int = 80000) -> Dict[int, List[str]]:
    return _fill_representatives(
        clusterer=clusterer, k=4,
        n_clusters=int(n_clusters_limit),
        boards_per_cluster=int(boards_per_cluster),
        seed=int(seed),
        max_samples=int(sample_pool),
    )

def discover_representative_rivers(*, clusterer, n_clusters_limit: int, boards_per_cluster: int, seed: int,
                                  sample_pool: int = 120000) -> Dict[int, List[str]]:
    return _fill_representatives(
        clusterer=clusterer, k=5,
        n_clusters=int(n_clusters_limit),
        boards_per_cluster=int(boards_per_cluster),
        seed=int(seed),
        max_samples=int(sample_pool),
    )
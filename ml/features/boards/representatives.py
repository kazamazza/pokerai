# ml/features/boards/representatives.py
from __future__ import annotations

from typing import Dict, List, Iterable, Tuple, Optional
import random

# If you already have a central card util, feel free to replace these
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

def _deck() -> List[str]:
    return [r + s for r in _RANKS for s in _SUITS]

def _sample_board_k(rng: random.Random, k: int) -> List[str]:
    """Sample k distinct cards from a fresh deck."""
    return rng.sample(_deck(), k)

def _board_to_str(cards: Iterable[str]) -> str:
    """Compact board encoding: '9dAc6h', '9dAc6hTh', '9dAc6hTh3s'."""
    # Ensure stable two-char tokens concatenated
    return "".join(cards)

def _yield_sampled_boards(
    *,
    rng: random.Random,
    k: int,
    sample_pool: int,
) -> Iterable[str]:
    """
    Yield up to sample_pool random legal boards of size k as compact strings.
    No attempt to deduplicate across generations; upstream caller dedups per cluster.
    """
    for _ in range(max(0, sample_pool)):
        yield _board_to_str(_sample_board_k(rng, k))

def _discover_representatives_generic(
    *,
    clusterer,
    k: int,                                # 3 flop, 4 turn, 5 river
    n_clusters_limit: int,
    boards_per_cluster: int,
    seed: int,
    sample_pool: int,
) -> Dict[int, List[str]]:
    """
    Generic sampler used by flop/turn/river discoverers.
    Returns: {cluster_id: [board_str, ...]}
    """
    rng = random.Random(int(seed))
    max_per_cluster = max(1, int(boards_per_cluster))
    target_clusters = max(1, int(n_clusters_limit))

    pools: Dict[int, List[str]] = {i: [] for i in range(target_clusters)}
    filled = 0
    # We iterate once through a sampled pool; for larger coverage, increase sample_pool in cfg.
    for board in _yield_sampled_boards(rng=rng, k=k, sample_pool=sample_pool):
        # Optional: fast skip if every cluster is already full
        if filled >= target_clusters * max_per_cluster:
            break

        # Predict cluster id (assumed 0..n-1). If predict returns floats/strings, coerce to int.
        try:
            cid_raw = clusterer.predict([board])[0]
        except Exception:
            # If your clusterer expects spaced boards like "9d Ac 6h", adapt here:
            spaced = " ".join([board[i:i+2] for i in range(0, len(board), 2)])
            cid_raw = clusterer.predict([spaced])[0]
        try:
            cid = int(cid_raw)
        except Exception:
            # Last-resort hashing in case the clusterer returns weird labels.
            cid = abs(hash(str(cid_raw))) % target_clusters

        if cid < 0 or cid >= target_clusters:
            # Guard against out-of-range cluster labels
            cid = cid % target_clusters

        bucket = pools[cid]
        # Dedup per cluster; cap at max_per_cluster
        if len(bucket) < max_per_cluster and board not in bucket:
            bucket.append(board)
            if len(bucket) == max_per_cluster:
                filled += max_per_cluster  # counted as filled to accelerate the early-stop

    # Remove empty clusters (optional; caller can tolerate empties if preferred)
    pools = {cid: boards for cid, boards in pools.items() if len(boards) > 0}
    return pools


# -----------------------------
# Public helpers (imported by ETL)
# -----------------------------
def discover_representative_flops(
    *,
    clusterer,
    n_clusters_limit: int,
    boards_per_cluster: int,
    seed: int,
    sample_pool: int = 50000,
) -> Dict[int, List[str]]:
    """
    Return representative flop boards by cluster: each board is a 6-char string (3 cards).
    Example: {'0': ['9dAc6h', ...], '1': [...], ...}  (keys are ints)
    """
    return _discover_representatives_generic(
        clusterer=clusterer,
        k=3,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

def discover_representative_turns(
    *,
    clusterer,
    n_clusters_limit: int,
    boards_per_cluster: int,
    seed: int,
    sample_pool: int = 80000,
) -> Dict[int, List[str]]:
    """
    Return representative turn boards by cluster: 8-char strings (4 cards).
    """
    return _discover_representatives_generic(
        clusterer=clusterer,
        k=4,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

def discover_representative_rivers(
    *,
    clusterer,
    n_clusters_limit: int,
    boards_per_cluster: int,
    seed: int,
    sample_pool: int = 120000,
) -> Dict[int, List[str]]:
    """
    Return representative river boards by cluster: 10-char strings (5 cards).
    """
    return _discover_representatives_generic(
        clusterer=clusterer,
        k=5,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )
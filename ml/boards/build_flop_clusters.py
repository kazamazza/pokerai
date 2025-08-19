# ml/boards/build_flop_clusters.py
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ml.utils.constants import R2I, S2I


# -----------------------------
# Canonicalization primitives
# -----------------------------

def card_key(cs: str) -> Tuple[int,int]:
    """Return sort key for a card string like 'Ah' using our fixed rank/suit order."""
    return (R2I[cs[0]], S2I[cs[1]])

def canon_flop_key(cards3: List[str]) -> str:
    """
    Deterministic canonical string for three cards.
    Sort by (rank_index, suit_index) using the SAME order we’ll use at runtime.
    """
    a,b,c = sorted(cards3, key=card_key)  # stable total order
    return a + b + c  # e.g., 'AhKdQc' (concatenated)

# -----------------------------
# Feature recipe (lite_v1, 18d)
# -----------------------------
def _rank_idxs(cards: List[str]) -> List[int]:
    return sorted([R2I[c[0]] for c in cards])  # 0..12 (0=Ace high)

def _suit_hist(cards: List[str]) -> List[int]:
    hist = [0,0,0,0]
    for c in cards:
        hist[S2I[c[1]]] += 1
    return hist  # [♠,♥,♦,♣]

def flop_features(board3: List[str]) -> List[float]:
    """
    18-dim cheap texture vector (no equities, fast):
      [is_pair, is_trips,
       min_gap/12, max_gap/12, sum_gap/(2*12),
       broadway/3, wheel, mono, two_tone, rainbow,
       hi3/3, lo3/3,
       spread/12, midness/(12/2),
       suit_hist/3 (4 dims)]
    """
    assert len(board3) == 3 and all(len(c)==2 for c in board3)
    r = _rank_idxs(board3)        # ascending index (A high => 0)
    s = _suit_hist(board3)

    is_pair  = int(r[0]==r[1] or r[1]==r[2] or r[0]==r[2])
    is_trips = int(r[0]==r[1]==r[2])

    gaps = [r[i+1]-r[i] for i in range(2)]
    min_gap, max_gap = min(gaps), max(gaps)
    sum_gap = sum(gaps)

    # “Broadway” density: A,K,Q,J,T → indices {0..4}
    broadway = sum(1 for x in r if x <= 4)
    # Wheel involvement heuristic: includes A and low run coverage
    wheel = int(any(idx in r for idx in [0,8,9,10,11,12]))

    s3 = int(3 in s)             # monotone
    s2 = int(2 in s)             # two-tone
    s1 = int(s.count(1) == 3)    # rainbow

    hi3 = sum(1 for x in r if x <= 4)  # same as broadway
    lo3 = sum(1 for x in r if x >= 8)

    spread  = r[-1]-r[0]
    midness = abs(r[1] - (r[0]+r[-1])/2.0)

    N = 12.0  # max index diff
    f = [
        is_pair, is_trips,
        min_gap/ N, max_gap/ N, sum_gap/(2*N),
        broadway/3.0, wheel,
        s3, s2, s1,
        hi3/3.0, lo3/3.0,
        spread/ N, midness/(N/2),
        s[0]/3.0, s[1]/3.0, s[2]/3.0, s[3]/3.0,
    ]
    return f

# -----------------------------
# Flop enumeration (all 22,100)
# -----------------------------
def enumerate_unique_flops() -> List[str]:
    """
    Produce all 3-card combos from the 52-card deck (no permutations).
    Return canonical keys (using the same canon we’ll use at runtime).
    Result length: 22,100.
    """
    deck = [r+s for r in RANKS for s in SUITS]
    flops = []
    L = len(deck)
    for i in range(L):
        for j in range(i+1, L):
            for k in range(j+1, L):
                a,b,c = deck[i], deck[j], deck[k]
                flops.append(canon_flop_key([a,b,c]))
    return flops

def flop_to_list(flop_key: str) -> List[str]:
    """Inverse of canon key → ['Ah','Kd','Qc']"""
    return [flop_key[0:2], flop_key[2:4], flop_key[4:6]]

# -----------------------------
# Build + persist clusters
# -----------------------------
def build_and_save_clusters(out_path: Path, K: int, seed: int, method_name: str = "lite_v1"):
    flops = enumerate_unique_flops()  # already canonical keys
    X = np.array([flop_features(flop_to_list(f)) for f in flops], dtype=np.float32)

    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(X)

    mapping: Dict[str,int] = {f: int(c) for f, c in zip(flops, labels)}
    out = {
        "meta": {
            "street": "flop",
            "k": K,
            "method": method_name,
            "seed": seed,
            "features": "lite_v1(18d)",
            "n_items": len(flops),
            "centroids": km.cluster_centers_.tolist()
        },
        "clusters": mapping
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"✅ wrote {out_path} | K={K} | items={len(flops)}")

# -----------------------------
# Self-check (miss-rate should be 0.0%)
# -----------------------------
def load_cluster_map(path: Path) -> Tuple[Dict[str,int], Dict]:
    obj = json.loads(path.read_text())
    return obj["clusters"], obj["meta"]

def sample_runtime_flop_key() -> str:
    """
    Simulate how your generators create boards:
    - sample three cards via a deck
    - convert to strings
    - use the SAME canon_flop_key used in enumeration/build
    """
    deck = [r+s for r in RANKS for s in SUITS]
    cards = random.sample(deck, 3)
    return canon_flop_key(cards)

def quick_self_check(out_path: Path, trials: int = 1000):
    clusters, meta = load_cluster_map(out_path)
    misses = 0
    for _ in range(trials):
        k = sample_runtime_flop_key()
        if k not in clusters:
            misses += 1
    miss_rate = 100.0 * misses / max(1, trials)
    print(f"Self-check (runtime lookup miss rate over {trials} random flops): {miss_rate:.4f}%")
    assert misses == 0, "Some runtime flops didn’t match cluster keys; check canonicalization."

# -----------------------------
# CLI
# -----------------------------
def main(settings_path="ml/config/settings.yaml"):
    import yaml
    cfg_all = yaml.safe_load(Path(settings_path).read_text())
    cfg     = cfg_all["board_clustering"]["flop"]

    K     = int(cfg["k"])
    seed  = int(cfg.get("seed", 42))
    out_p = Path(cfg["out_path"])

    build_and_save_clusters(out_p, K=K, seed=seed, method_name=cfg.get("method","lite_v1"))
    quick_self_check(out_p, trials=1000)

if __name__ == "__main__":
    random.seed(42)
    main()
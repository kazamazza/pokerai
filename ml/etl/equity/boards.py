import random
from typing import List, Tuple
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def all_flops() -> List[Tuple[str,str,str]]:
    deck = [r+s for r in RANKS for s in SUITS]
    flops = []
    for i in range(len(deck)):
        for j in range(i+1,len(deck)):
            for k in range(j+1,len(deck)):
                r1,r2,r3 = deck[i], deck[j], deck[k]
                flops.append((r1,r2,r3))
    return flops  # 22100 combos

def flop_features(flop: Tuple[str,str,str]) -> List[float]:
    # Simple, fast features (extend later if needed)
    rs = sorted([RANKS.index(c[0]) for c in flop], reverse=True)  # hi→lo idx
    suits = [c[1] for c in flop]
    suit_counts = Counter(suits).values()
    is_paired = int(len({c[0] for c in flop}) < 3)
    is_monotone = int(max(suit_counts) == 3)
    is_two_tone = int(max(suit_counts) == 2)
    gap1 = rs[0]-rs[1]; gap2 = rs[1]-rs[2]
    broadway = sum(int(RANKS[i] in "TJQKA") for i in rs)
    return [rs[0], rs[1], rs[2], is_paired, is_monotone, is_two_tone, gap1, gap2, broadway]

def build_flop_clusters(k=256, seed=42):
    from sklearn.cluster import KMeans
    flops = all_flops()
    X = [flop_features(f) for f in flops]
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    assign = km.labels_.tolist()
    # index flops by cluster for fast sampling
    clusters = [[] for _ in range(k)]
    for f, cid in zip(flops, assign):
        clusters[cid].append(f)
    return km, clusters  # keep km if you want centroids later

def sample_flops_from_cluster(clusters, cid: int, m: int, dead: set[str] | None=None):
    dead = dead or set()
    pool = [f for f in clusters[cid] if not (set(f) & dead)]
    if not pool:
        return []
    if m >= len(pool):
        random.shuffle(pool)
        return pool
    return random.sample(pool, m)
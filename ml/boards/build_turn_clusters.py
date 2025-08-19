# ml/boards/build_turn_clusters.py
from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ml.utils.constants import R2I, SUITS


# --- basic card utils (keep synced with your central utils if you move them) ---


def _rank_idxs(cards: List[str]) -> List[int]:
    return sorted([R2I[c[0]] for c in cards])

def _suit_hist(cards: List[str]) -> List[int]:  # [n♠,n♥,n♦,n♣]
    h = [0,0,0,0]
    for c in cards:
        h[SUITS.index(c[1])] += 1
    return h

def flop_features(board3: List[str]) -> List[float]:
    """Same lite_v1 flop 18d features you used earlier."""
    assert len(board3) == 3 and all(len(c)==2 for c in board3)
    r = _rank_idxs(board3)
    s = _suit_hist(board3)

    is_pair  = int(r[0]==r[1] or r[1]==r[2] or r[0]==r[2])
    is_trips = int(r[0]==r[1]==r[2])

    gaps = [r[i+1]-r[i] for i in range(2)]
    min_gap = min(gaps); max_gap = max(gaps); sum_gap = sum(gaps)

    broadway = sum(1 for x in r if x <= 4)             # A,K,Q,J,T present
    wheel    = int(any(idx in r for idx in [0,8,9,10,11,12]))  # A or low wheel-ish ranks

    s3 = int(3 in s)          # monotone
    s2 = int(2 in s)          # two-tone
    s1 = int(s.count(1)==3)   # rainbow

    hi3 = sum(1 for x in r if x <= 4)
    lo3 = sum(1 for x in r if x >= 8)

    spread  = r[-1]-r[0]
    midness = abs(r[1] - (r[0]+r[-1])/2.0)

    N = 12.0
    return [
        is_pair, is_trips,
        min_gap/N, max_gap/N, sum_gap/(2*N),
        broadway/3.0, wheel,
        s3, s2, s1,
        hi3/3.0, lo3/3.0,
        spread/N, midness/(N/2),
        s[0]/3.0, s[1]/3.0, s[2]/3.0, s[3]/3.0,
    ]  # 18 dims

def turn_features(board4: List[str]) -> List[float]:
    """
    Lite v1 turn texture features (fast, deterministic).
    = flop_features(first 3) + turn deltas/signals (10 dims) => 28 dims total.
    """
    assert len(board4) == 4 and all(len(c)==2 for c in board4)
    flop3 = board4[:3]
    turn  = board4[3]
    ft = flop_features(flop3)  # 18d

    r4 = _rank_idxs(board4)
    s4 = _suit_hist(board4)

    # Pair/trips/quads status on turn
    # count occurrences per distinct rank
    from collections import Counter
    rc = Counter([c[0] for c in board4])
    counts = sorted(rc.values(), reverse=True)  # e.g., [2,1,1] or [3,1] etc.
    is_any_pair  = int(2 in counts or 3 in counts or 4 in counts)
    is_trips_now = int(3 in counts)
    is_quads_now = int(4 in counts)

    # Did the turn *pair* the flop? (i.e., created/added a pair relative to flop)
    rc3 = Counter([c[0] for c in flop3])
    turn_rank = turn[0]
    paired_turn = int(rc3[turn_rank] >= 1)

    # Suit/flush signals
    max_suit = max(s4)               # 1..4
    two_tone_turn   = int(2 in s4)
    monotone_turn   = int(3 in s4)
    four_flush_turn = int(4 in s4)   # all same suit on turn (rare but representable)

    # Straightiness with 4 cards (max run length among ranks A..2 treating adjacency)
    # Simple proxy: span & number of distinct ranks
    distinct_ranks = sorted({R2I[c[0]] for c in board4})
    span = distinct_ranks[-1] - distinct_ranks[0] if len(distinct_ranks) >= 2 else 0
    n_dist = len(distinct_ranks)
    # “4-run” potential if span <= 3 over >=4 ranks (very connected)
    four_run_potential = int(n_dist == 4 and span <= 3)

    # High/low density with 4 cards
    hi4 = sum(1 for x in r4 if x <= 4) / 4.0
    lo4 = sum(1 for x in r4 if x >= 8) / 4.0

    # Assemble 10 new dims
    N = 12.0
    add10 = [
        is_any_pair, is_trips_now, is_quads_now,
        paired_turn,
        two_tone_turn, monotone_turn, four_flush_turn,
        four_run_potential,
        span / N,
        hi4,  # already normalized
        # lo4 we’ll include too to keep symmetry
        lo4,
    ]
    # That’s 11 — keep 10 by dropping either hi4 or lo4; keep both → 29 dims total. Let's keep both.
    # Final dims: 18 + 11 = 29
    return ft + add10  # 29 dims

def rand_turn() -> List[str]:
    deck = [r+s for r in RANKS for s in SUITS]
    return random.sample(deck, 4)

def assign_cluster(feat: np.ndarray, centroids: np.ndarray) -> int:
    # nearest centroid by L2
    d = ((centroids - feat)**2).sum(axis=1)
    return int(np.argmin(d))

# ------------------------------ main ------------------------------
def main(settings_path="ml/config/settings.yaml"):
    import yaml
    cfg_all = yaml.safe_load(Path(settings_path).read_text())
    cfg     = cfg_all["board_clustering"]["turn"]

    K            = int(cfg.get("k", 256))
    seed         = int(cfg.get("seed", 42))
    fit_sample_n = int(cfg.get("fit_sample_n", 300_000))
    out_p        = Path(cfg.get("out_path", "data/boards/turn_clusters.k256.lite_v1.json"))

    random.seed(seed)
    np.random.seed(seed)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fitting turn clusters: K={K} | mode=centroids | seed={seed} | fit_sample_n={fit_sample_n}")

    # Sample turns and featurize
    feats = []
    for _ in range(fit_sample_n):
        t = rand_turn()
        feats.append(turn_features(t))
    X = np.asarray(feats, dtype=np.float32)

    # KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(X)

    obj = {
        "meta": {
            "street": "turn",
            "k": K,
            "method": "lite_v1",
            "seed": seed,
            "features": "lite_v1(29d)",
            "fit_sample_n": fit_sample_n,
            "centroids": km.cluster_centers_.tolist()  # centroids only
        }
        # note: no explicit {"clusters": ...} map; we assign by nearest centroid at runtime
    }
    out_p.write_text(json.dumps(obj))
    print(f"✅ wrote {out_p} (centroids only)")

    # Self-check: how many distinct clusters do random turns hit?
    centroids = np.asarray(km.cluster_centers_, dtype=np.float32)
    hits = set()
    for _ in range(2000):
        f = np.asarray(turn_features(rand_turn()), dtype=np.float32)
        cid = assign_cluster(f, centroids)
        hits.add(cid)
    print(f"Self-check: assigned 2000 random turns → K={len(hits)} clusters hit")

if __name__ == "__main__":
    main()
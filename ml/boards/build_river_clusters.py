import json, gzip, random, math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# ---- shared card constants (reuse your ml.types.cards if present) ----
try:
    from ml.utils.cards import RANKS, SUITS, R2I
except Exception:
    RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
    SUITS = ['s','h','d','c']
    R2I = {r:i for i,r in enumerate(RANKS)}  # A=0 … 2=12

# ---- deterministic board key: keep dealt order (same convention as flop) ----
def canon_board_key(cards: List[str]) -> str:
    """['Ah','Kd','Qc','2s','9h'] -> 'AhKdQc2s9h'"""
    return "".join(cards)

# ---- lightweight, permutation-invariant river features (~28 dims) ----
def _suit_hist(cards: List[str]) -> List[int]:
    idx = {s:i for i,s in enumerate(SUITS)}
    h = [0,0,0,0]
    for c in cards: h[idx[c[1]]] += 1
    return h

def _rank_hist(cards: List[str]) -> List[int]:
    h = [0]*13
    for c in cards: h[R2I[c[0]]] += 1
    return h

def _max_consecutive(ranks_uniq: List[int]) -> int:
    if not ranks_uniq: return 0
    best = run = 1
    for i in range(1, len(ranks_uniq)):
        if ranks_uniq[i] == ranks_uniq[i-1] + 1:
            run += 1; best = max(best, run)
        else:
            run = 1
    return best

def river_features(board5: List[str]) -> List[float]:
    """
    Fast, purely combinatorial texture vector for 5-card board.
    No equities—suitable for clustering millions of rivers quickly.
    """
    assert len(board5) == 5 and all(len(c)==2 for c in board5)

    # histograms
    s_hist = _suit_hist(board5)        # 4 dims (sum=5)
    r_hist = _rank_hist(board5)        # 13 dims (sum=5)

    # multiplicities
    pairs   = sum(1 for x in r_hist if x==2)
    trips   = sum(1 for x in r_hist if x==3)
    quads   = int(any(x==4 for x in r_hist))
    two_pair= int(pairs>=2)
    full    = int( (3 in r_hist) and (2 in r_hist) )

    # straight-ish: work on unique rank indices high=A(0)..low=2(12)
    uniq = sorted(i for i,x in enumerate(r_hist) if x>0)
    max_run = _max_consecutive(uniq)
    wheel  = int(0 in uniq and 9 in uniq and 10 in uniq and 11 in uniq and 12 in uniq)  # A-5 heuristic
    broadway = int(all(i in uniq for i in [0,1,2,3,4]))  # A-K-Q-J-T present

    # suit texture flags
    s5 = int(5 in s_hist)   # mono (flush board)
    s4 = int(4 in s_hist)   # 4-to-flush
    s3 = int(3 in s_hist)   # 3-suited
    s2 = int(s_hist.count(2) == 2)  # two pairs of suits
    s1 = int(s_hist.count(1) >= 3)  # 3+ rainbow singles

    # normalize simple scalars
    N = 12.0
    # spread and midness (using rank indices replicated by multiplicity)
    ranks_rep = []
    for i,x in enumerate(r_hist):
        ranks_rep.extend([i]*x)
    spread  = (max(ranks_rep) - min(ranks_rep)) / N
    midness = abs(np.median(ranks_rep) - (min(ranks_rep)+max(ranks_rep))/2.0) / (N/2.0)

    # pack features
    f = [
        # suit histogram normalized
        s_hist[0]/5.0, s_hist[1]/5.0, s_hist[2]/5.0, s_hist[3]/5.0,
        # rank multiplicities
        pairs, trips, quads, two_pair, full,
        # straight-ish
        max_run/5.0, wheel, broadway,
        # suit texture flags
        s5, s4, s3, s2, s1,
        # spread & middle-ness
        spread, midness,
    ]
    # append coarse rank-shape summary: top-3 counts sorted desc (adds stability)
    top3 = sorted(r_hist, reverse=True)[:3]
    f.extend(top3)  # sum ≤ 5
    return f  # ~28 dims

# ---- enumerators (full set and sampler) ----
def iter_all_rivers() -> Iterable[List[str]]:
    """Yield every 5-card board as list of strings (e.g., ['Ah','Kd','Qc','2s','9h'])."""
    deck = [r+s for r in RANKS for s in SUITS]  # 52 cards
    L = len(deck)
    for a in range(L):
        for b in range(a+1, L):
            for c in range(b+1, L):
                for d in range(c+1, L):
                    for e in range(d+1, L):
                        yield [deck[a], deck[b], deck[c], deck[d], deck[e]]

def sample_rivers(n: int, seed: int = 42) -> List[List[str]]:
    """Uniform sample without replacement (approx) by random shuffles."""
    random.seed(seed)
    deck = [r+s for r in RANKS for s in SUITS]
    out, seen = [], set()
    while len(out) < n:
        random.shuffle(deck)
        board = deck[:5]
        key = canon_board_key(board)
        if key in seen: continue
        seen.add(key)
        out.append(board[:])
    return out

# ---- I/O helpers ----
def write_meta(out_path: Path, meta: Dict, clusters: Dict[str,int] | None, centroids: np.ndarray):
    obj = {
        "meta": meta | {
            "centroids": centroids.tolist(),
        },
        "clusters": clusters if clusters is not None else {}
    }
    out_path.write_text(json.dumps(obj))

def write_full_map_gz(map_path: Path, mapping: Dict[str,int], meta: Dict, centroids: np.ndarray):
    """
    Write a single JSON with meta+centroids+clusters (gzipped) without loading all into RAM.
    """
    # Dump in two steps into a gzipped file: header then clusters object in chunks
    with gzip.open(map_path, "wt", encoding="utf-8") as f:
        f.write('{"meta":')
        f.write(json.dumps(meta | {"centroids": centroids.tolist()}))
        f.write(',"clusters":{')
        first = True
        for k,v in mapping.items():
            if not first: f.write(',')
            first = False
            # keys are small, safe to json.dumps
            f.write(json.dumps(k)); f.write(':'); f.write(str(int(v)))
        f.write('}}')

# ---- main ----
def main(settings_path="ml/config/settings.yaml"):
    import yaml
    cfg_all = yaml.safe_load(Path(settings_path).read_text())
    cfg     = cfg_all["board_clustering"]["river"]

    K        = int(cfg["k"])
    seed     = int(cfg.get("seed", 42))
    method   = cfg.get("method", "lite_v1")
    mode     = cfg.get("mode", "centroids")        # "centroids" or "full_map"
    fit_n    = int(cfg.get("fit_sample_n", 400_000))  # used to fit KMeans
    out_p    = Path(cfg["out_path"])               # e.g., data/boards/river_clusters.k256.lite_v1.json
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1) Fit KMeans on a sample (MiniBatch for speed/memory)
    print(f"Fitting river clusters: K={K} | mode={mode} | seed={seed} | fit_sample_n={fit_n}")
    fit_boards = sample_rivers(fit_n, seed=seed)
    X_fit = np.array([river_features(b) for b in fit_boards], dtype=np.float32)

    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=5)
    km.fit(X_fit)

    meta = {
        "street": "river",
        "k": K, "method": method, "seed": seed,
        "features": "lite_v1(~28d)",
        "fit_sample_n": fit_n
    }

    if mode == "centroids":
        # 2a) Save centroids-only model (small, no misses at runtime)
        write_meta(out_p, meta, clusters=None, centroids=km.cluster_centers_)
        print(f"✅ wrote {out_p} (centroids only)")

        # Self-check: assign 2k random rivers; no 'miss' concept (we compute features and predict)
        test_boards = sample_rivers(2000, seed=seed+1)
        X_test = np.array([river_features(b) for b in test_boards], dtype=np.float32)
        preds = km.predict(X_test)
        print(f"Self-check: assigned 2000 random rivers → K={len(set(preds))} clusters hit")
        return

    # 2b) FULL MAP (heavy): assign every 5-card board and store as gzipped JSON
    map_gz = Path(str(out_p) + ".gz")
    print("Assigning all rivers to clusters (this can take a while)…")
    mapping: Dict[str,int] = {}
    batch: List[Tuple[str, List[float]]] = []
    batch_size = 100_000
    total = 0

    for board in iter_all_rivers():
        key = canon_board_key(board)
        batch.append((key, river_features(board)))
        if len(batch) >= batch_size:
            X = np.array([f for _,f in batch], dtype=np.float32)
            labels = km.predict(X)
            for (k,_), lab in zip(batch, labels):
                mapping[k] = int(lab)
            total += len(batch)
            print(f"  assigned {total:,} rivers…")
            batch.clear()

    if batch:
        X = np.array([f for _,f in batch], dtype=np.float32)
        labels = km.predict(X)
        for (k,_), lab in zip(batch, labels):
            mapping[k] = int(lab)
        total += len(batch)
        batch.clear()

    write_full_map_gz(map_gz, mapping, meta, km.cluster_centers_)
    print(f"✅ wrote {map_gz} (clusters map gz) | entries={total:,}")

    # Self-check (map presence): sample 1000 random rivers and ensure keys exist
    misses = 0
    with gzip.open(map_gz, "rt", encoding="utf-8") as f:
        data = json.loads(f.read())   # (only safe for moderate sizes; for very large files, skip this check)
        clusters = data["clusters"]
    for b in sample_rivers(1000, seed=seed+2):
        if canon_board_key(b) not in clusters:
            misses += 1
    print(f"Self-check (key presence over 1000 samples): miss_rate={misses/1000:.4f}")
    assert misses == 0, "Some runtime rivers didn’t match cluster keys."

if __name__ == "__main__":
    main()
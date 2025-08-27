import sys
from pathlib import Path
import json, random
from typing import List
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.config.types_hands import RANKS, SUITS
from ml.features.boards.board_features import featurize_board


def sample_flops(n: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    deck = [r+s for r in RANKS for s in SUITS]
    boards = []
    for _ in range(n):
        b = random.sample(deck, 3)  # flop example
        boards.append("".join(b))
    return boards

def main():
    import argparse
    from sklearn.cluster import KMeans

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="artifact path (json)")
    ap.add_argument("--n_clusters", type=int, default=246)
    ap.add_argument("--n_samples", type=int, default=22100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) Sample boards
    boards = sample_flops(args.n_samples, args.seed)

    # 2) Featurize to numeric vectors (NOT BoardFeatures objects)
    #    featurize_board(...) should return a BoardFeatures with .to_vector()
    feats = []
    for b in boards:
        f = featurize_board(b)
        # if featurize_board already returns a numpy array, this will still work
        v = f.to_vector() if hasattr(f, "to_vector") else np.asarray(f, dtype=np.float32)
        feats.append(np.asarray(v, dtype=np.float32))
    X = np.stack(feats, axis=0)  # shape [N, D], float32

    # 3) Fit KMeans
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init="auto").fit(X)

    # 4) Write JSON artifact with centroids
    art = {
        "kind": "kmeans",
        "n_clusters": int(args.n_clusters),
        "random_state": int(args.seed),
        "centroids": km.cluster_centers_.astype(float).tolist(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art))
    print(f"✅ wrote kmeans artifact → {out_path}")

if __name__ == "__main__":
    main()
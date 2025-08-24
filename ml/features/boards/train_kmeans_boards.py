# ml/features/boards/train_kmeans_boards.py
from pathlib import Path
import json, random
from typing import List
import numpy as np

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
    ap.add_argument("--n_clusters", type=int, default=48)
    ap.add_argument("--n_samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    boards = sample_flops(args.n_samples, args.seed)
    X = np.stack([featurize_board(b) for b in boards], axis=0)

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init="auto").fit(X)
    art = {
        "kind": "kmeans",
        "n_clusters": int(args.n_clusters),
        "random_state": int(args.seed),
        "centroids": km.cluster_centers_.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(art))
    print(f"✅ wrote kmeans artifact → {args.out}")

if __name__ == "__main__":
    main()
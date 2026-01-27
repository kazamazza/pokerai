from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

import json
import random
from pathlib import Path
from typing import List, Literal

import numpy as np
from sklearn.cluster import KMeans

from ml.config.types_hands import RANKS, SUITS
from ml.features.boards.board_features import featurize_board

StreetName = Literal["flop", "turn", "river"]


def _deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def sample_boards(*, n: int, k: int, seed: int) -> List[str]:
    """
    Sample n random boards with k distinct cards, returned as canonical concatenated strings.
    Deterministic via local RNG (does NOT mutate global random state).
    """
    rng = random.Random(int(seed))
    deck = _deck()
    out: List[str] = []
    for _ in range(int(n)):
        cards = rng.sample(deck, k)
        out.append("".join(cards))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--street", type=str, required=True, choices=["flop", "turn", "river"])
    ap.add_argument("--out", type=str, required=True, help="artifact path (json)")
    ap.add_argument("--n_clusters", type=int, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_init", type=int, default=10)  # stable across sklearn versions
    args = ap.parse_args()

    street: StreetName = args.street  # type: ignore
    k = {"flop": 3, "turn": 4, "river": 5}[street]

    boards = sample_boards(n=args.n_samples, k=k, seed=args.seed)

    feats = []
    for b in boards:
        v = np.asarray(featurize_board(b).to_vector(), dtype=np.float32)
        feats.append(v)
    X = np.stack(feats, axis=0)

    km = KMeans(
        n_clusters=int(args.n_clusters),
        random_state=int(args.seed),
        n_init=int(args.n_init),
    ).fit(X)

    art = {
        "kind": "kmeans_board_clusters",
        "street": street,
        "n_cards": k,
        "n_clusters": int(args.n_clusters),
        "random_state": int(args.seed),
        "feature_dim": int(km.cluster_centers_.shape[1]),
        "centroids": km.cluster_centers_.astype(np.float32).tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art))
    print(f"✅ wrote kmeans artifact → {out_path} (street={street}, K={args.n_clusters}, D={art['feature_dim']})")


if __name__ == "__main__":
    main()
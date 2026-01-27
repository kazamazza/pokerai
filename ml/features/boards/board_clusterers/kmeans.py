from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from ml.features.boards.board_features import featurize_board


class KMeansBoardClusterer:
    """
    JSON centroid artifact ONLY (portable, deterministic).

    Artifact schema:
      {
        "kind": "kmeans_board_clusters",
        "street": "flop"|"turn"|"river",
        "n_cards": 3|4|5,
        "n_clusters": K,
        "feature_dim": D,
        "centroids": [[...D...], ...K rows...]
      }
    """

    def __init__(self, *, n_clusters: int, n_cards: int, feature_dim: int, centroids: np.ndarray):
        self.n_clusters = int(n_clusters)
        self.n_cards = int(n_cards)
        self.feature_dim = int(feature_dim)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        if self.centroids.shape != (self.n_clusters, self.feature_dim):
            raise ValueError(
                f"centroids shape mismatch: expected {(self.n_clusters, self.feature_dim)} got {self.centroids.shape}"
            )

    @staticmethod
    def load(path: str | Path) -> "KMeansBoardClusterer":
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError(f"KMeansBoardClusterer expects a .json artifact, got: {path}")

        meta = json.loads(path.read_text())
        if not isinstance(meta, dict) or meta.get("kind") != "kmeans_board_clusters":
            raise ValueError(f"Bad artifact kind in {path}: {meta.get('kind')}")

        n_clusters = int(meta["n_clusters"])
        n_cards = int(meta["n_cards"])
        feature_dim = int(meta["feature_dim"])
        centroids = np.asarray(meta["centroids"], dtype=np.float32)

        return KMeansBoardClusterer(
            n_clusters=n_clusters,
            n_cards=n_cards,
            feature_dim=feature_dim,
            centroids=centroids,
        )

    def _featurize(self, board: str) -> np.ndarray:
        # strict: ensure board length matches n_cards
        exp_len = self.n_cards * 2
        if not board or len(board) != exp_len:
            raise ValueError(f"Board '{board}' length {len(board) if board else 0} != expected {exp_len}")

        v = np.asarray(featurize_board(board).to_vector(), dtype=np.float32)
        if v.shape != (self.feature_dim,):
            raise ValueError(f"Feature dim mismatch: got {v.shape} expected ({self.feature_dim},)")
        return v

    def predict(self, boards: List[str]) -> List[int]:
        X = np.stack([self._featurize(b) for b in boards], axis=0)  # [N,D]

        # nearest centroid (squared euclidean)
        X2 = (X ** 2).sum(axis=1, keepdims=True)           # [N,1]
        C2 = (self.centroids ** 2).sum(axis=1)[None, :]    # [1,K]
        XC = X @ self.centroids.T                           # [N,K]
        d2 = X2 - 2.0 * XC + C2                             # [N,K]

        labels = d2.argmin(axis=1)
        return [int(i) for i in labels.tolist()]

    def predict_one(self, board: str) -> int:
        return self.predict([board])[0]
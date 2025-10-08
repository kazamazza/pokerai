import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None  # optional, only needed for fitting or pickle artifacts

from ml.features.boards.board_features import featurize_board  # assumes .to_vector()


class KMeansBoardClusterer:
    """
    Supports two artifact styles:
      1) Pickled sklearn KMeans model (legacy).
      2) JSON with precomputed centroids (recommended for portability).

    Predict will use model.predict if model exists; otherwise nearest-centroid on stored centroids.
    """
    def __init__(self, n_clusters: int = 20, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: Optional["KMeans"] = None
        self.centroids: Optional[np.ndarray] = None  # shape [K, D]

    def fit(self, boards: List[str]) -> None:
        if KMeans is None:
            raise RuntimeError("scikit-learn is required to fit KMeansBoardClusterer")
        X = np.array([featurize_board(b).to_vector() for b in boards], dtype=np.float32)
        self.model = KMeans(n_clusters=self.n_clusters, n_init="auto",
                            random_state=self.random_state)
        self.model.fit(X)
        self.centroids = np.asarray(self.model.cluster_centers_, dtype=np.float32)

    def _predict_nearest_centroid(self, X: np.ndarray) -> List[int]:
        if self.centroids is None:
            raise RuntimeError("No centroids available for prediction")
        X2 = (X ** 2).sum(axis=1, keepdims=True)         # [N,1]
        C2 = (self.centroids ** 2).sum(axis=1, keepdims=True).T  # [1,K]
        XC = X @ self.centroids.T                        # [N,K]
        d2 = X2 - 2 * XC + C2                            # [N,K]
        labels = d2.argmin(axis=1)
        return [int(i) for i in labels.tolist()]

    def predict(self, boards: List[str]) -> List[int]:
        if self.model is None and self.centroids is None:
            raise RuntimeError("KMeansBoardClusterer not initialized (no model or centroids)")
        X = np.array([featurize_board(b).to_vector() for b in boards], dtype=np.float32)
        if self.model is not None:
            labels = self.model.predict(X)
            return [int(i) for i in labels.tolist()]
        return self._predict_nearest_centroid(X)

    def predict_one(self, board: str) -> int:
        return self.predict([board])[0]

    def save(self, path: str | Path) -> None:
        """
        Legacy: saves pickle with sklearn model if present; else saves JSON with centroids.
        """
        path = Path(path)
        if self.model is not None:
            with path.open("wb") as f:
                pickle.dump({
                    "type": "kmeans",
                    "n_clusters": self.n_clusters,
                    "random_state": self.random_state,
                    "model": self.model
                }, f)
        else:
            if self.centroids is None:
                raise RuntimeError("Nothing to save (no model and no centroids)")
            payload = {
                "kind": "kmeans",
                "n_clusters": int(self.n_clusters),
                "random_state": int(self.random_state),
                "centroids": self.centroids.tolist(),
            }
            path.with_suffix(".json").write_text(json.dumps(payload))

    @staticmethod
    def load(path: str | Path) -> "KMeansBoardClusterer":
        """
        Loads either:
          - JSON artifact: {"kind":"kmeans","n_clusters":...,"centroids":[...]}
          - Pickle artifact with sklearn model.
        """
        path = Path(path)
        # Try JSON first
        try:
            if path.suffix.lower() == ".json":
                meta = json.loads(path.read_text())
            else:
                # Some pipelines may still point to a .pkl that actually contains JSON (rare).
                # Try JSON parse optimistically.
                try:
                    meta = json.loads(path.read_text())
                except Exception:
                    meta = None
            if isinstance(meta, dict) and "centroids" in meta:
                obj = KMeansBoardClusterer(
                    n_clusters=int(meta.get("n_clusters", 0) or 0),
                    random_state=int(meta.get("random_state", 42)),
                )
                obj.centroids = np.asarray(meta["centroids"], dtype=np.float32)
                if obj.n_clusters == 0:
                    obj.n_clusters = int(obj.centroids.shape[0])
                return obj
        except Exception:
            pass  # fall through to pickle

        # Fallback: pickle with sklearn KMeans
        with path.open("rb") as f:
            meta = pickle.load(f)
        obj = KMeansBoardClusterer(n_clusters=meta.get("n_clusters", 0),
                                   random_state=meta.get("random_state", 42))
        obj.model = meta.get("model", None)
        if obj.model is not None and hasattr(obj.model, "cluster_centers_"):
            obj.centroids = np.asarray(obj.model.cluster_centers_, dtype=np.float32)
            if obj.n_clusters == 0:
                obj.n_clusters = int(obj.centroids.shape[0])
        return obj
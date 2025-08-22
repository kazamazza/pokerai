import pickle
from typing import Optional, List
import numpy as np
from sklearn.cluster import KMeans
from ml.features.boards.board_features import featurize_board


class KMeansBoardClusterer:
    """
    Vector-based clusterer over BoardFeatures using KMeans.
    """
    def __init__(self, n_clusters: int = 20, random_state: int = 42):
        if np is None or KMeans is None:
            raise RuntimeError("scikit-learn/numpy required for KMeansBoardClusterer")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: Optional[KMeans] = None

    def fit(self, boards: List[str]) -> None:
        X = np.array([featurize_board(b).to_vector() for b in boards], dtype=np.float32)
        self.model = KMeans(n_clusters=self.n_clusters, n_init="auto",
                            random_state=self.random_state)
        self.model.fit(X)

    def predict(self, boards: List[str]) -> List[int]:
        if self.model is None:
            raise RuntimeError("KMeansBoardClusterer not fit")
        X = np.array([featurize_board(b).to_vector() for b in boards], dtype=np.float32)
        labels = self.model.predict(X)
        return [int(i) for i in labels.tolist()]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "type": "kmeans",
                "n_clusters": self.n_clusters,
                "random_state": self.random_state,
                "model": self.model
            }, f)

    @staticmethod
    def load(path: str) -> "KMeansBoardClusterer":
        with open(path, "rb") as f:
            meta = pickle.load(f)
        obj = KMeansBoardClusterer(n_clusters=meta["n_clusters"],
                                   random_state=meta["random_state"])
        obj.model = meta["model"]
        return obj
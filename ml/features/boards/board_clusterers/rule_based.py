import pickle
from typing import List, Sequence, Optional
from ml.config.types_hands import RANK_TO_I
from ml.features.boards.board_features import featurize_board


class RuleBasedBoardClusterer:
    """
    Fast, interpretable bucketizer.
    """
    def __init__(self, n_clusters: Optional[int] = None, version: str = "v1") -> None:
        self.version = version
        self.n_clusters = n_clusters  # for protocol compatibility; rule-based may ignore

    def predict_one(self, board_str: str) -> int:
        f = featurize_board(board_str)
        if f.monotone: suit_bucket = 3
        elif f.has_3suited: suit_bucket = 2
        elif f.has_2suited: suit_bucket = 1
        else: suit_bucket = 0

        if f.quads: pair_bucket = 3
        elif f.trips: pair_bucket = 2
        elif f.paired: pair_bucket = 1
        else: pair_bucket = 0

        if f.connectivity <= 1.0: conn_bucket = 2
        elif f.connectivity <= 2.0: conn_bucket = 1
        else: conn_bucket = 0

        high_bucket = 1 if f.max_rank >= RANK_TO_I['Q'] else 0
        cid = suit_bucket * 24 + pair_bucket * 6 + conn_bucket * 2 + high_bucket
        return int(cid)

    def predict(self, boards: List[str]) -> List[int]:
        return [self.predict_one(b) for b in boards]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"type": "rule", "version": self.version, "n_clusters": self.n_clusters}, f)

    @staticmethod
    def load(path: str) -> "RuleBasedBoardClusterer":
        with open(path, "rb") as f:
            meta = pickle.load(f)
        return RuleBasedBoardClusterer(
            n_clusters=meta.get("n_clusters"),
            version=meta.get("version", "v1"),
        )
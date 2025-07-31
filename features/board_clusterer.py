import json
import os
from typing import List
from features.board_normalizer import BoardNormalizer
from features.types import PROJECT_ROOT


class BoardClusterer:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(PROJECT_ROOT, "data/board_textures.json")

        with open(path, "r") as f:
            self.clusters = json.load(f)

        self.key_to_id = {key: idx for idx, key in enumerate(self.clusters.keys())}
        self.normalizer = BoardNormalizer()

    def get_cluster_key(self, board: List[str]) -> str:
        return self.normalizer.normalize(board)

    def get_cluster_id(self, board: List[str]) -> int:
        key = self.normalizer.normalize(board)
        return self.key_to_id.get(key, -1)
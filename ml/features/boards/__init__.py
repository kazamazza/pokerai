from pathlib import Path
from typing import Dict, Any

from ml.features.boards.board_clusterers.kmeans import KMeansBoardClusterer
from ml.features.boards.board_clusterers.rule_based import RuleBasedBoardClusterer
from ml.features.boards.board_protocols import BoardClusterer


def load_board_clusterer(cfg: Dict[str, Any]) -> BoardClusterer:
    bc = cfg.get("board_clustering") or cfg.get("board_clustering")
    if not bc:
        raise KeyError("board_clustering section missing in cfg")

    kind = (bc.get("type") or "rule").lower()

    if kind == "rule":
        n = int(bc.get("n_clusters", 48))
        return RuleBasedBoardClusterer(n_clusters=n)

    if kind == "kmeans":
        art = bc.get("artifact")
        print(art)
        if not art:
            raise ValueError("kmeans clusterer requires 'artifact' path")
        return KMeansBoardClusterer.load(art)

    raise ValueError(f"Unknown board clusterer type: {kind}")
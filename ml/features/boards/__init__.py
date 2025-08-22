from ml.features.boards.board_clusterers.kmeans import KMeansBoardClusterer
from ml.features.boards.board_clusterers.rule_based import RuleBasedBoardClusterer


def load_board_clusterer(cfg):
    kind = cfg["board_clustering"]["type"]  # "rule" or "kmeans"
    if kind == "rule":
        return RuleBasedBoardClusterer()
    elif kind == "kmeans":
        art = cfg["board_clustering"]["artifact"]
        return KMeansBoardClusterer.load(art)  # your load method
    else:
        raise ValueError(f"Unknown board clustering type: {kind}")
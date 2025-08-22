from pathlib import Path

from ml.features.boards import load_board_clusterer
from ml.features.boards.board_clusterers.kmeans import KMeansBoardClusterer
from ml.features.boards.board_clusterers.rule_based import RuleBasedBoardClusterer


# ---------- Rule-based clusterer tests ----------

def test_rule_clusterer_predict_basic():
    cfg = {"board_clustering": {"type": "rule"}}
    clusterer = load_board_clusterer(cfg)

    boards = ["AhKdQs", "2c3c4c", "AsAd7h", "9d9c9s"]
    labels = clusterer.predict(boards)

    assert isinstance(clusterer, RuleBasedBoardClusterer)
    assert len(labels) == len(boards)

def test_rule_clusterer_determinism():
    cfg = {"board_clustering": {"type": "rule"}}
    clusterer = load_board_clusterer(cfg)

    boards = ["AhKdQs", "2c3c4c", "AsAd7h", "9d9c9s", "Td8d7d"]
    labels1 = clusterer.predict(boards)
    labels2 = clusterer.predict(boards)

    assert labels1 == labels2  # pure function, no randomness

def test_rule_clusterer_batch_consistency():
    cfg = {"board_clustering": {"type": "rule"}}
    clusterer = load_board_clusterer(cfg)

    boards = ["AhKdQs", "2c3c4c", "AsAd7h", "9d9c9s", "Td8d7d", "2h2d2c"]
    # predict together
    all_labels = clusterer.predict(boards)
    # predict individually and compare
    single_labels = [clusterer.predict([b])[0] for b in boards]

    assert all_labels == single_labels


# ---------- K-Means clusterer tests ----------

def _fit_and_save_kmeans(tmp_path: Path, n_clusters: int = 5, random_state: int = 0) -> Path:
    # small synthetic set spanning different textures
    train_boards = [
        "AhKdQs", "KhQdJc",         # high broadway / dry-ish
        "2c3c4c", "5h6h7h",         # monotone / connected
        "AsAd7h", "KcKd2d",         # paired
        "9d9c9s", "2h2d2c",         # trips (boards with pairs/trips)
        "AhTh2h", "Qc9c3c",         # two-tone / monotone mix
        "AcKcQc", "JsTs9s",         # strong monotone runouts
        "7d6c5h", "6s5s4d",         # low connected
    ]

    km = KMeansBoardClusterer(n_clusters=n_clusters, random_state=random_state)
    km.fit(train_boards)
    out_path = tmp_path / "kmeans_boards.pkl"
    km.save(out_path)
    return out_path


def test_kmeans_fit_predict_roundtrip(tmp_path):
    art_path = _fit_and_save_kmeans(tmp_path, n_clusters=4, random_state=123)

    cfg = {"board_clustering": {"type": "kmeans", "artifact": str(art_path)}}
    clusterer = load_board_clusterer(cfg)

    boards = ["AhKdQs", "2c3c4c", "AsAd7h", "9d9c9s", "Td8d7d"]
    labels = clusterer.predict(boards)

    assert isinstance(clusterer, KMeansBoardClusterer)
    assert len(labels) == len(boards)


def test_kmeans_deterministic_after_load(tmp_path):
    art_path = _fit_and_save_kmeans(tmp_path, n_clusters=6, random_state=7)
    cfg = {"board_clustering": {"type": "kmeans", "artifact": str(art_path)}}
    clusterer = load_board_clusterer(cfg)

    boards = ["AhKdQs", "2c3c4c", "AsAd7h", "9d9c9s", "Td8d7d", "AcKcQc"]
    labels1 = clusterer.predict(boards)
    labels2 = clusterer.predict(boards)

    assert labels1 == labels2  # prediction should be deterministic given fixed model


def test_kmeans_handles_unseen_boards(tmp_path):
    art_path = _fit_and_save_kmeans(tmp_path, n_clusters=5, random_state=0)
    cfg = {"board_clustering": {"type": "kmeans", "artifact": str(art_path)}}
    clusterer = load_board_clusterer(cfg)

    # Boards not present in training set (but same 3-card format)
    unseen = ["Ad4d8d", "QcJs9h", "2s5d7c", "AhKhTh", "3c3s7d"]
    labels = clusterer.predict(unseen)

    assert len(labels) == len(unseen)
    # labels must be integer cluster ids
    assert all(isinstance(l, int) for l in labels)
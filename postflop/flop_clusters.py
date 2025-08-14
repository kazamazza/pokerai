import json
from pathlib import Path
from typing import Dict, Tuple, Set

FLOP_CLUSTER_MAP_PATH = Path("data/flop/flop_cluster_map.json")

def load_flop_cluster_map() -> Dict[str, int]:
    with FLOP_CLUSTER_MAP_PATH.open("r") as f:
        return json.load(f)

def valid_clusters_and_reps(cluster_map: Dict[str, int]) -> Tuple[Set[int], Dict[int, str]]:
    valid = set(cluster_map.values())
    rep: Dict[int, str] = {}
    for board, cid in cluster_map.items():
        if cid not in rep:
            rep[cid] = board
    return valid, rep

def board_for_cluster_id(cid: int, rep_map: Dict[int, str]) -> str:
    return rep_map[cid]
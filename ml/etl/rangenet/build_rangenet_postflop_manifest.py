from __future__ import annotations
import argparse, random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.config.types_hands import RANKS, SUITS
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config


def _all_52() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]

def _random_flop(rng: random.Random) -> List[str]:
    deck = _all_52()
    return rng.sample(deck, k=3)

def _compact(board_cards: List[str]) -> str:
    # ["Ah","Kd","7c"] -> "AhKd7c"
    return "".join(board_cards)

def _discover_cluster_ids(clusterer, sample_n: int, seed: int) -> List[int]:
    """Probe cluster ids that actually appear."""
    rng = random.Random(seed)
    ids: List[int] = []
    for _ in range(sample_n):
        b = _compact(_random_flop(rng))
        ids.append(clusterer.predict([b])[0])
    # keep frequency order (most common first)
    from collections import Counter
    cnt = Counter(ids)
    return [cid for cid, _ in cnt.most_common()]

def _sample_boards_for_cluster(clusterer, target_cluster_id: int, n_boards: int, seed: int) -> List[str]:
    """Find n concrete flops that map to the given cluster id."""
    rng = random.Random(seed)
    out: List[str] = []
    attempts = 0
    max_attempts = n_boards * 600  # safeguard
    while len(out) < n_boards and attempts < max_attempts:
        b = _compact(_random_flop(rng))
        lbl = clusterer.predict([b])[0]
        if lbl == target_cluster_id and b not in out:
            out.append(b)
        attempts += 1
    return out

def build_manifest(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Emit one row per (stack_bb, pot_bb, ip_pos, oop_pos, actor, street, board_cluster_id, board, node_key).
    """
    # ---- read knobs ----
    seed = int(cfg.get("seed", 42))
    rb = cfg.get("rangenet_postflop", {})
    mb = rb.get("manifest_build", {})
    stacks_bb: List[int] = list(mb.get("stacks_bb", [40, 80]))
    pots_bb:   List[int] = list(mb.get("pot_bb", [6, 12]))
    streets:   List[int] = list(mb.get("streets", [1]))  # 1=flop (start here)
    pos_pairs: List[Dict[str,str]] = list(mb.get("position_pairs", [{"ip":"BTN","oop":"BB"}]))
    actors:    List[str] = list(mb.get("actors", ["ip"]))
    node_keys: List[str] = list(mb.get("node_keys", ["flop_root"]))
    n_clusters_limit = int(mb.get("board_clusters_limit", 24))
    boards_per_cluster = int(mb.get("boards_per_cluster", 3))
    dedupe = bool(mb.get("dedupe", True))

    # ---- board clustering & cluster discovery ----
    clusterer = load_board_clusterer(cfg)
    # discover actually used cluster ids, then keep the top-N
    discovered = _discover_cluster_ids(clusterer, sample_n=2000, seed=seed)
    cluster_ids = discovered[:n_clusters_limit] if n_clusters_limit > 0 else discovered

    # build a small pool of boards per cluster
    cid_to_boards: Dict[int, List[str]] = {}
    for cid in cluster_ids:
        boards = _sample_boards_for_cluster(clusterer, cid, boards_per_cluster, seed=seed + cid * 101)
        if boards:
            cid_to_boards[cid] = boards

    if not cid_to_boards:
        raise RuntimeError("Could not sample any boards for clusters; check clusterer/config.")

    # ---- cartesian product over scenario knobs ----
    rows: List[Dict[str, Any]] = []
    for street in streets:
        if street != 1:
            # You can extend to turn/river later (needs more board handling)
            continue
        for stack in stacks_bb:
            for pot in pots_bb:
                for pp in pos_pairs:
                    ip_pos, oop_pos = str(pp["ip"]), str(pp["oop"])
                    for actor in actors:          # "ip" or "oop"
                        for node_key in node_keys:
                            for cid, boards in cid_to_boards.items():
                                for b in boards:
                                    rows.append({
                                        "stack_bb": int(stack),
                                        "pot_bb": float(pot),
                                        "ip_pos": ip_pos,
                                        "oop_pos": oop_pos,
                                        "actor": actor,               # whose strategy we’ll read at node_key
                                        "street": int(street),        # 1=flop
                                        "board_cluster_id": int(cid),
                                        "board": b,                   # compact string "AhKd7c"
                                        "node_key": node_key,
                                        "weight": float(1.0),         # simple weight; tune later if needed
                                    })

    df = pd.DataFrame(rows)
    if dedupe and not df.empty:
        df = df.drop_duplicates(
            subset=["stack_bb","pot_bb","ip_pos","oop_pos","actor","street","board_cluster_id","board","node_key"],
            keep="first"
        ).reset_index(drop=True)

    return df

def run_from_config(cfg_path_or_name: str) -> Path:
    cfg = load_model_config(cfg_path_or_name)
    out_path = Path(
        cfg.get("rangenet_postflop", {})
           .get("outputs", {})
           .get("manifest", "data/artifacts/rangenet_postflop_manifest.parquet")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_manifest(cfg)
    df.to_parquet(out_path, index=False)
    print(f"✅ wrote manifest → {out_path}  rows={len(df):,}")
    print("   Schema: stack_bb:int, pot_bb:float, ip_pos:str, oop_pos:str, actor:str, "
          "street:int, board_cluster_id:int, board:str, node_key:str, weight:float")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rangenet_postflop",
                    help="Model name or YAML path (resolved by load_model_config)")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional override for outputs.manifest")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.out:
        cfg.setdefault("rangenet_postflop", {}).setdefault("outputs", {})["manifest"] = args.out
    # Save back? not required; just pass to runner
    run_from_config(args.config)

if __name__ == "__main__":
    main()
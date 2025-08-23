from __future__ import annotations
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.features.equity.shared import expand_range_to_combos_weighted
from ml.config.types_hands import RANKS, SUITS, ALL_HANDS
from ml.etl.utils.monker_parser import load_range_file_cached
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config
from ml.features.equity.postflop import equity_postflop_vs_range_cached_combos


def sha1_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate_villain_range(manifest_rows: pd.DataFrame) -> Dict[str, float]:
    """
    Merge Monker ranges from all files in the subset (last non-empty line wins per file,
    then we average across files, then normalize).
    """
    acc: Dict[str, float] = defaultdict(float)
    n = 0
    for _, r in manifest_rows.iterrows():
        p = Path(r["abs_path"])
        rng_map = load_range_file_cached(p)
        if not rng_map:
            continue
        for k, v in rng_map.items():
            acc[k] += v
        n += 1

    if n == 0:
        return {}

    # Average then normalize
    for k in acc:
        acc[k] = acc[k] / n
    s = sum(acc.values())
    if s > 0:
        for k in acc:
            acc[k] /= s
    return dict(acc)


def _all_52() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]

def _random_flop(rng: random.Random) -> List[str]:
    """Draw 3 distinct random cards as strings, e.g. ['As','Kd','7h']"""
    deck = _all_52()
    return rng.sample(deck, k=3)

def _sample_boards_for_cluster(clusterer, target_cluster_id: int, n_boards: int, seed: int) -> List[List[str]]:
    """
    Naive sampler: generate random flops until we collect n_boards whose predicted cluster == target_cluster_id.
    """
    rng = random.Random(seed)
    out: List[List[str]] = []
    attempts = 0
    max_attempts = n_boards * 500  # safeguard
    while len(out) < n_boards and attempts < max_attempts:
        b = _random_flop(rng)
        lbl = clusterer.predict(["".join(b)])[0]  # predict expects ['AhKdQs'] format
        if lbl == target_cluster_id:
            out.append(b)
        attempts += 1
    return out

def _discover_cluster_to_boards(clusterer, n_clusters_limit: int, boards_per_cluster: int, seed: int) -> Dict[int, List[List[str]]]:
    """
    Discover up to `n_clusters_limit` cluster ids and collect `boards_per_cluster` boards for each.
    """
    # Try to discover cluster ids by sampling
    rng = random.Random(seed)
    seen: set[int] = set()
    # First, sample a pool of boards to see which cluster ids exist:
    pool: List[List[str]] = []
    labels: List[int] = []
    for _ in range(2000):
        b = _random_flop(rng)
        pool.append(b)
        labels.append(clusterer.predict(["".join(b)])[0])
    # Rank clusters by frequency and pick top `n_clusters_limit`
    freq: Dict[int, int] = defaultdict(int)
    for lab in labels:
        freq[lab] += 1
        seen.add(lab)
    cluster_ids = [cid for cid, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]
    if n_clusters_limit > 0:
        cluster_ids = cluster_ids[:n_clusters_limit]

    # Now fill boards per cluster
    out: Dict[int, List[List[str]]] = {}
    for cid in cluster_ids:
        boards = _sample_boards_for_cluster(clusterer, cid, boards_per_cluster, seed=seed + cid)
        if boards:
            out[cid] = boards
    return out


def compute_postflop_equity(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Dict[str, Any],
) -> None:
    manifest = pd.read_parquet(manifest_path)
    required = {"stack_bb", "hero_pos", "opener_action", "abs_path"}
    missing = sorted([c for c in required if c not in manifest.columns])
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Board clusterer
    clusterer = load_board_clusterer(cfg)
    # Config knobs (with safe defaults)
    bc_cfg = cfg.get("board_clustering", {})
    n_clusters_limit = int(bc_cfg.get("n_clusters_limit", 24))
    boards_per_cluster = int(bc_cfg.get("boards_per_cluster", 8))
    mc_cfg = cfg.get("equity_mc", {})
    n_samples_per_eval = int(mc_cfg.get("postflop_n_samples", 1500))
    seed = int(mc_cfg.get("seed", 42))

    # Discover a set of clusters + sample boards for each
    cluster_to_boards = _discover_cluster_to_boards(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
    )
    if not cluster_to_boards:
        raise RuntimeError("Could not discover any board clusters; check your clusterer/config.")

    rows: List[Dict[str, Any]] = []
    # Group manifest into scenarios
    gcols = ["stack_bb", "hero_pos", "opener_action"]
    for keys, df_g in manifest.groupby(gcols):
        stack_bb, hero_pos, opener_action = keys

        # 1) Aggregate villain range for this scenario (merges all Monker files)
        vill_range = aggregate_villain_range(df_g)
        if not vill_range:
            continue

        # 2) Pre-expand once per scenario into concrete combos + weights
        #    This avoids recomputing expansion & normalization on every call.
        vill_combos, vill_weights = expand_range_to_combos_weighted(vill_range)
        # (optional safety normalize)
        s = sum(vill_weights)
        if s > 0:
            vill_weights = [w / s for w in vill_weights]
        else:
            # if somehow empty, skip scenario
            continue

        # 3) For each hero hand and each cluster, average equity across sampled boards
        for hand_id, hero_code in enumerate(ALL_HANDS):
            for cluster_id, boards in cluster_to_boards.items():
                wins = ties = loses = 0.0

                for i, board_cards in enumerate(boards):
                    w, t, l = equity_postflop_vs_range_cached_combos(
                        board_cards=board_cards,
                        hero_code=hero_code,
                        vill_combos=vill_combos,
                        vill_weights=vill_weights,
                        n_samples=n_samples_per_eval,
                        seed=seed + i + hand_id * 17 + cluster_id * 101,
                    )
                    wins += w;
                    ties += t;
                    loses += l

                n = len(boards)
                if n == 0:
                    continue

                y_win = wins / n
                y_tie = ties / n
                y_lose = loses / n
                weight = float(n * n_samples_per_eval)

                rows.append({
                    "stack_bb": int(stack_bb),
                    "hero_pos": str(hero_pos),
                    "opener_action": str(opener_action),
                    "hand_id": int(hand_id),
                    "board_cluster_id": int(cluster_id),
                    "y_win": float(y_win),
                    "y_tie": float(y_tie),
                    "y_lose": float(y_lose),
                    "weight": weight,
                })

    out_df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"✅ wrote {out_parquet} with {len(out_df):,} rows")
    print("   Schema: stack_bb:int, hero_pos:str, opener_action:str, hand_id:int, "
          "board_cluster_id:int, y_win:float, y_tie:float, y_lose:float, weight:float")

def run_from_config(cfg: dict) -> None:
    def get(path: str, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    manifest_path = Path(get("inputs.manifest"))
    out_parquet   = Path(get("outputs.postflop_parquet", "data/datasets/equitynet_postflop.parquet"))

    if not manifest_path or not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    # The function already consumes cfg (board_clustering + equity_mc)
    compute_postflop_equity(
        manifest_path=manifest_path,
        out_parquet=out_parquet,
        cfg=cfg,
    )
    print(f"✅ wrote {out_parquet}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="equitynet_postflop",
                    help="YAML config with inputs/outputs/board_clustering/equity_mc")
    # Optional one-off override for output path
    ap.add_argument("--out", type=str, default=None,
                    help="(Optional) override outputs.postflop_parquet")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.out:
        cfg.setdefault("outputs", {})["postflop_parquet"] = args.out

    run_from_config(cfg)


if __name__ == "__main__":
    main()
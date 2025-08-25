from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.utils import hand_range_to_vec169, avg_vecs
from ml.features.boards import load_board_clusterer
from ml.range.solvers.adapter_cached import load_villain_range_cached_only
from ml.utils.config import load_model_config


def _row_to_solver_params(r: pd.Series, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Build the canonical param dict used by the solver workers for this row.
    Returns (params, err) where err is non-empty if we cannot build params.
    """
    # Pull solver defaults
    solv = cfg.get("solver", {})
    accuracy         = float(solv.get("accuracy", 0.5))
    max_iter         = int(solv.get("max_iter", 200))
    allin_threshold  = float(solv.get("allin_threshold", 0.67))
    bet_sizing_id    = str(solv.get("bet_sizing_id", "std"))
    use_clusters     = bool(solv.get("use_board_clusters", True))

    # Required scenario bits (manifest should carry these)
    try:
        street  = int(r.get("street", 1))  # 1=flop, 2=turn, 3=river
        stack   = float(r["stack_bb"])
    except Exception:
        return {}, "missing_stack_or_street"

    pot = r.get("pot_bb", None)
    if pot is None:
        # If pot not present, you can choose to skip or derive a default.
        # Safer: skip — the key won’t match pre-solved outputs without the same pot.
        return {}, "missing_pot_bb"

    # Positions: prefer explicit "positions" column ("OOPvIP"/"IPvOOP")
    positions = r.get("positions", None)
    if not positions:
        # Derive from villain_pos if you store "IP"/"OOP" there
        vill_pos = str(r.get("villain_pos", "")).upper()
        if vill_pos in ("IP", "OOP"):
            positions = f"{vill_pos}v{'IP' if vill_pos=='OOP' else 'OOP'}"
        else:
            return {}, "missing_positions"

    # Ranges (must match the worker’s input strings)
    range_ip  = r.get("range_ip", None)
    range_oop = r.get("range_oop", None)
    if not range_ip or not range_oop:
        return {}, "missing_ranges"

    # Board: exact or clustered
    board_str = str(r.get("board_str", "")).strip()
    bc_id     = int(r.get("board_cluster_id", -1))
    if use_clusters:
        if bc_id < 0:
            return {}, "missing_board_cluster"
        board = None
        board_cluster_id = bc_id
    else:
        if not board_str:
            return {}, "missing_board_str"
        board = board_str
        board_cluster_id = None

    params = {
        "street": street,
        "pot_bb": float(pot),
        "effective_stack_bb": float(stack),
        "board": board,
        "board_cluster_id": board_cluster_id,
        "range_ip": str(range_ip),
        "range_oop": str(range_oop),
        "positions": positions,
        "bet_sizing_id": bet_sizing_id,
        "accuracy": accuracy,
        "max_iter": max_iter,
        "allin_threshold": allin_threshold,
    }
    return params, ""

def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Dict[str, Any],
) -> None:
    """
    Cache-only builder:
      - Reads your postflop manifest
      - Computes (board_cluster_id) if needed
      - For each row, constructs canonical solver params
      - Loads pre-solved JSON from cache/S3
      - Aggregates to scenario rows with y_0..y_168 and weight
    """
    df = pd.read_parquet(manifest_path)

    # Normalize villain_pos if manifest used opener_pos
    if "villain_pos" not in df.columns and "opener_pos" in df.columns:
        df = df.rename(columns={"opener_pos": "villain_pos"})

    # Ensure board strings & clusters exist (only if using clusters)
    bc_cfg = cfg.get("board_clustering", {})
    use_clusters = bool(cfg.get("solver", {}).get("use_board_clusters", True))
    if use_clusters:
        if "board_cluster_id" not in df.columns:
            # Compute clusters here (expects compact like "AhKd7c")
            clusterer = load_board_clusterer(cfg)
            df["board_str"] = (
                df.get("board_str")
                  if "board_str" in df.columns
                  else df["board"].astype(str).str.replace(" ", "", regex=False)
            )
            df["board_cluster_id"] = clusterer.predict(df["board_str"].tolist())
    else:
        # Ensure board_str present for exact-board keys
        if "board_str" not in df.columns:
            df["board_str"] = df["board"].astype(str).str.replace(" ", "", regex=False)

    # Stable node_key (if you have a helper; otherwise keep a placeholder)
    if "node_key" not in df.columns:
        df["node_key"] = "flop_root"  # or infer_node_key(df row)

    # Grouping columns for final aggregation
    gcols = ["stack_bb", "hero_pos", "villain_pos", "street", "board_cluster_id", "node_key"] \
            if use_clusters else \
            ["stack_bb", "hero_pos", "villain_pos", "street", "board_str", "node_key"]

    rows: List[Dict[str, Any]] = []
    buckets: Dict[Tuple, List[np.ndarray]] = {}
    weights: Dict[Tuple, float] = {}

    # Track skip reasons
    skipped: Dict[str, int] = {}

    for _, r in df.iterrows():
        params, err = _row_to_solver_params(r, cfg)
        if err:
            skipped[err] = skipped.get(err, 0) + 1
            continue

        # Decide whose range to extract at this node (villain)
        # If villain is IP in positions "IPvOOP", actor="ip", else "oop"
        pos = params["positions"]
        vill = str(r.get("villain_pos", "")).upper()
        if vill in ("IP", "OOP"):
            actor = "ip" if (vill == "IP") else "oop"
        else:
            # fallback from positions string
            actor = "ip" if pos.startswith("IP") else "oop"

        try:
            rng_map = load_villain_range_cached_only(
                cfg=cfg,
                pot_bb=params["pot_bb"],
                effective_stack_bb=params["effective_stack_bb"],
                board=params["board"],
                board_cluster_id=params["board_cluster_id"],
                range_ip=params["range_ip"],
                range_oop=params["range_oop"],
                positions=params["positions"],
                street=params["street"],
                bet_sizing_id=params["bet_sizing_id"],
                accuracy=params["accuracy"],
                max_iter=params["max_iter"],
                allin_threshold=params["allin_threshold"],
                actor=actor,
                node_key=str(r["node_key"]),
                local_cache_dir=cfg.get("solver", {}).get("local_cache_dir", "data/solver_cache"),
            )
        except FileNotFoundError:
            skipped["cache_miss"] = skipped.get("cache_miss", 0) + 1
            continue

        if not rng_map:
            skipped["empty_parse"] = skipped.get("empty_parse", 0) + 1
            continue

        y = hand_range_to_vec169(rng_map)

        key = tuple(r[c] for c in gcols)
        buckets.setdefault(key, []).append(y)
        weights[key] = weights.get(key, 0.0) + 1.0

    # Build output rows
    for key, y_list in buckets.items():
        y_avg = avg_vecs(y_list)
        row = {
            "stack_bb": int(key[0]),
            "hero_pos": str(key[1]),
            "villain_pos": str(key[2]),
            "street": int(key[3]),
            "node_key": str(key[-1]),
            "weight": float(weights.get(key, len(y_list))),
        }
        if use_clusters:
            row["board_cluster_id"] = int(key[4])
        else:
            row["board_str"] = str(key[4])

        for i, val in enumerate(y_avg.tolist()):
            row[f"y_{i}"] = float(val)

        rows.append(row)

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(f"✅ wrote {out_parquet} with {len(out):,} rows")
    if skipped:
        tot = sum(skipped.values())
        print("   Skipped rows:", ", ".join(f"{k}={v}" for k, v in skipped.items()), f"(total {tot})")
    if not len(out):
        print("   (No rows — likely cache misses or manifest lacks ranges/positions/pot)")

def run_from_config(cfg: Dict[str, Any]) -> None:
    """
    Config shape (example):

    rangenet_postflop:
      inputs:
        manifest: data/artifacts/solver_manifest.parquet
      outputs:
        parquet: data/datasets/rangenet_postflop.parquet
    board_clustering:
      type: rule   # or kmeans (+ artifact)
    """
    def get(path: str, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    manifest = Path(get("inputs.manifest"))
    out_pq   = Path(get("outputs.parquet", "data/datasets/rangenet_postflop.parquet"))

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    build_rangenet_postflop(manifest, out_pq, cfg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # as with your other scripts: pass a short name or a yaml path
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="Model name or YAML path (resolved by load_model_config)")
    # optional one-off override for output
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.out:
        cfg.setdefault("rangenet_postflop", {}).setdefault("outputs", {})["parquet"] = args.out

    run_from_config(cfg)


if __name__ == "__main__":
    main()
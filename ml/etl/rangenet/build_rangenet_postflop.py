from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from ml.etl.rangenet.utils import hand_range_to_vec169, avg_vecs, infer_node_key
from ml.features.boards import load_board_clusterer
from ml.range.solvers.adapter import load_villain_range_from_solver
from ml.utils.config import load_model_config


def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Dict[str, Any],
) -> None:
    df = pd.read_parquet(manifest_path)
    req = {"stack_bb", "hero_pos", "street", "board", "abs_path"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Normalize villain position column name if your manifest uses opener_pos
    if "villain_pos" not in df.columns and "opener_pos" in df.columns:
        df = df.rename(columns={"opener_pos": "villain_pos"})

    # board clustering (shared with Equity work)
    clusterer = load_board_clusterer(cfg)
    # predict expects compact strings like "AhKd7c"
    df["board_str"] = df["board"].astype(str).str.replace(" ", "", regex=False)
    df["board_cluster_id"] = clusterer.predict(df["board_str"].tolist())

    # Produce a stable node_key per row
    df["node_key"] = df.apply(infer_node_key, axis=1)

    # Vectorize each file’s villain range
    vecs: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    # We’ll aggregate by scenario keys:
    gcols = ["stack_bb", "hero_pos", "villain_pos", "street", "board_cluster_id", "node_key"]

    # Collect vectors per group for averaging
    buckets: Dict[Tuple, List[np.ndarray]] = {}
    weights: Dict[Tuple, float] = {}

    for _, r in df.iterrows():
        key = tuple(r[c] for c in gcols)
        solver_path = Path(str(r["abs_path"]))
        node_key = r["node_key"]

        rng_map = load_villain_range_from_solver(solver_path, node_key=node_key)
        if not rng_map:
            # skip empty parses
            continue

        y = hand_range_to_vec169(rng_map)
        buckets.setdefault(key, []).append(y)
        weights[key] = weights.get(key, 0.0) + 1.0  # simple count; tune if you have per-file weight

    # Build rows
    for key, y_list in buckets.items():
        y_avg = avg_vecs(y_list)
        row = {
            "stack_bb": int(key[0]),
            "hero_pos": str(key[1]),
            "villain_pos": str(key[2]),
            "street": int(key[3]),
            "board_cluster_id": int(key[4]),
            "node_key": str(key[5]),
            "weight": float(weights.get(key, len(y_list))),
        }
        # attach y_0..y_168
        for i, val in enumerate(y_avg.tolist()):
            row[f"y_{i}"] = float(val)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(f"✅ wrote {out_parquet} with {len(out):,} rows")
    if not len(out):
        print("   (No rows — check manifest paths / solver adapter)")

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

    manifest = Path(get("rangenet_postflop.inputs.manifest"))
    out_pq   = Path(get("rangenet_postflop.outputs.parquet", "data/datasets/rangenet_postflop.parquet"))

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    build_rangenet_postflop(manifest, out_pq, cfg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # as with your other scripts: pass a short name or a yaml path
    ap.add_argument("--config", type=str, default="rangenet_postflop",
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
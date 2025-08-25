from __future__ import annotations
import itertools
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.features.boards.board_clusterers.utils import discover_representative_flops
from ml.range.solvers.keying import solve_sha1, s3_key_for_solve
from ml.range.solvers.utils.preflop_ranges import get_ranges_for_pair
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config


def _get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def build_manifest(cfg: dict) -> pd.DataFrame:
    # -------- read sections with fallbacks --------
    rpf = _get(cfg, "rangenet_postflop", {}) or {}
    mb  = rpf.get("manifest_build", {}) or {}

    # Board clustering settings can live under rangenet_postflop.board_clustering
    # or top-level board_clustering — support both.
    bc_cfg = rpf.get("board_clustering", None)
    if bc_cfg is None:
        bc_cfg = _get(cfg, "board_clustering", {}) or {}

    # Solver knobs
    sv = rpf.get("solver", {})
    acc   = float(sv.get("accuracy", 0.75))
    iters = int(sv.get("max_iterations", 100))
    a_th  = float(sv.get("allin_threshold", 0.67))
    ver   = str(sv.get("version", "v1"))
    s3_prefix = str(sv.get("s3_prefix", f"solver/outputs/{ver}"))

    # -------- manifest_build knobs (with safe defaults) --------
    stacks  = [float(x) for x in mb.get("stacks_bb", [100])]
    pots    = [float(x) for x in mb.get("pots_bb", [20])]           # ✅ default so it won’t crash
    streets = [int(x)   for x in mb.get("streets", [1])]            # 1=flop
    position_pairs = [tuple(x) for x in mb.get("position_pairs", [["IP","OOP"]])]
    bet_menu_ids   = [str(x)   for x in mb.get("bet_menus", ["std"])]

    # Board sampling parameters
    n_clusters = int(mb.get("board_clusters_limit", mb.get("board_clusters", 24)))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool = int(mb.get("sample_pool", 20000))
    seed = int(_get(cfg, "seed", 42))

    # -------- board clusterer + representative flops --------
    clusterer = load_board_clusterer(cfg)  # respects bc_cfg under the hood
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

    # -------- build rows --------
    rows: List[Dict[str, Any]] = []
    for street in streets:
        for stack, pot in itertools.product(stacks, pots):
            for (ip, oop) in position_pairs:
                # Preflop-derived ranges for this stack/positions (Monker-style compact strings)
                rng_ip, rng_oop = get_ranges_for_pair(stack_bb=stack, ip=ip, oop=oop, cfg=cfg)

                for menu in bet_menu_ids:
                    for cluster_id, boards in boards_by_cluster.items():
                        # For dev we include each representative board as a separate solve
                        for b in boards:
                            board_str = "".join(b)  # e.g., "QsJh2h"
                            params = {
                                "street": street,
                                "pot_bb": pot,
                                "effective_stack_bb": stack,
                                "board": board_str,
                                "board_cluster_id": cluster_id,
                                "range_ip": rng_ip,
                                "range_oop": rng_oop,
                                "positions": f"{ip}v{oop}",
                                "bet_sizing_id": menu,
                                "accuracy": acc,
                                "max_iter": iters,
                                "allin_threshold": a_th,
                                "solver_version": ver,
                            }
                            sha = solve_sha1(params)
                            s3k = s3_key_for_solve(params, sha1=sha, prefix=s3_prefix)

                            rows.append({
                                **params,
                                "sha1": sha,
                                "s3_key": s3k,
                                "node_key": "root",
                                "weight": 1.0,
                            })

    df = pd.DataFrame(rows)
    return df

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rangenet/postflop", help="model[/variant]/profile")
    ap.add_argument("--out", type=str, default="data/artifacts/rangenet_postflop_manifest.parquet")
    args = ap.parse_args()

    cfg = load_model_config(model=args.config)  # your resolver supports model/variant/profile
    df = build_manifest(cfg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote manifest: {out} rows={len(df):,}")

if __name__ == "__main__":
    main()
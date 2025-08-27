from __future__ import annotations
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

def build_flop_manifest(cfg: dict) -> pd.DataFrame:
    """
    Flop-only manifest builder for RangeNet Postflop.
    Produces rows with street=1 and representative flops per board-cluster.
    """
    rpf = _get(cfg, "rangenet_postflop", {}) or {}
    mb  = rpf.get("manifest_build", {}) or {}

    # Solver knobs
    sv = rpf.get("solver", {}) or {}
    acc   = float(sv.get("accuracy", 0.75))
    iters = int(sv.get("max_iterations", 100))
    a_th  = float(sv.get("allin_threshold", 0.67))
    ver   = str(sv.get("version", "v1"))
    s3_prefix = str(sv.get("s3_prefix", f"solver/outputs/{ver}"))

    # Manifest knobs (flop only: NO streets in config)
    stacks  = [float(x) for x in mb.get("stacks_bb", [100])]
    pots    = [float(x) for x in mb.get("pots_bb",   [20])]
    position_pairs = [tuple(x) for x in mb.get("position_pairs", [("BTN","BB")])]
    bet_menu_ids   = [str(x)   for x in mb.get("bet_menus", ["std"])]

    n_clusters_limit   = int(mb.get("board_clusters_limit", 24))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool        = int(mb.get("sample_pool", 20000))
    seed               = int(_get(cfg, "seed", 42))

    # Board clusterer + representative flops
    clusterer = load_board_clusterer(cfg)
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

    rows: List[Dict[str, Any]] = []
    street = 1  # FLOP (fixed)

    for stack in stacks:
        for pot in pots:
            for (ip_pos, oop_pos) in position_pairs:
                # Resolve preflop ranges (compact strings) for this pair & stack
                rng_ip, rng_oop = get_ranges_for_pair(
                    stack_bb=stack,
                    ip=ip_pos,
                    oop=oop_pos,
                    cfg=cfg
                )

                for menu in bet_menu_ids:
                    for cluster_id, boards in boards_by_cluster.items():
                        for b in boards:
                            board_str = "".join(b)  # e.g., "QsJh2h"
                            params = {
                                "street": street,  # fixed to FLOP
                                "pot_bb": pot,
                                "effective_stack_bb": stack,
                                "board": board_str,
                                "board_cluster_id": int(cluster_id),
                                "range_ip": rng_ip,
                                "range_oop": rng_oop,
                                "positions": f"{ip_pos}v{oop_pos}",
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

    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Build flop-only manifest for RangeNet Postflop"
    )
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="model[/variant]/profile")
    ap.add_argument("--out", type=str,
                    default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    args = ap.parse_args()

    cfg = load_model_config(model=args.config)
    df = build_flop_manifest(cfg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote FLOP manifest: {out} rows={len(df):,}")
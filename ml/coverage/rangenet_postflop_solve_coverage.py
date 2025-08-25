import sys

import pandas as pd
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.parser import try_load_solved_from_cache


def solve_coverage_from_manifest(
    manifest_parquet: str | Path,
    cfg: Dict[str, Any],
    local_cache_dir: str | Path = "data/solver_cache",
) -> Dict[str, Any]:
    man = pd.read_parquet(manifest_parquet)
    # one row per planned solve with the params you wrote in build_manifest
    # expected columns: street,pot_bb,effective_stack_bb,board,board_cluster_id,range_ip,range_oop,
    #                   positions,bet_sizing_id,accuracy,max_iter,allin_threshold,sha1,s3_key,weight,node_key
    total = len(man)
    solved_flags = []

    for _, r in man.iterrows():
        params = {
            "street": int(r["street"]),
            "pot_bb": float(r["pot_bb"]),
            "effective_stack_bb": float(r["effective_stack_bb"]),
            "board": r.get("board"),
            "board_cluster_id": int(r["board_cluster_id"]),
            "range_ip": r["range_ip"],
            "range_oop": r["range_oop"],
            "positions": r["positions"],
            "bet_sizing_id": r["bet_sizing_id"],
            "accuracy": float(r["accuracy"]),
            "max_iter": int(r["max_iter"]),
            "allin_threshold": float(r["allin_threshold"]),
        }
        hit = try_load_solved_from_cache(cfg, params, Path(local_cache_dir))
        solved_flags.append(hit is not None)

    man = man.copy()
    man["solved"] = solved_flags

    gcols = ["effective_stack_bb","pot_bb","positions","street","board_cluster_id"]
    agg = (man.groupby(gcols, as_index=False)
             .agg(planned=("solved","size"),
                  solved=("solved","sum")))
    agg["pct"] = (agg["solved"] / agg["planned"]) * 100.0

    summary = {
        "rangenet_postflop_solve": {
            "planned": int(total),
            "solved": int(man["solved"].sum()),
            "pct": float(100.0 * man["solved"].mean() if total else 0.0),
        }
    }
    return {"summary": summary, "by_bucket": agg.to_dict(orient="records")}

def main():
    import argparse, json
    from ml.utils.config import load_model_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)   # rangenet_postflop (dev/prod)
    ap.add_argument("--manifest", required=True) # manifest parquet
    ap.add_argument("--out", default="reports/coverage/rangenet_postflop_solve.json")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    cov = solve_coverage_from_manifest(args.manifest, cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cov, indent=2))
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()
# tools/probe_solver_files.py
import sys, pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor

def main(manifest_path: str, local_solver_dir: str, sample_n: int = 20):
    df = pd.read_parquet(manifest_path).head(sample_n)
    x = TexasSolverExtractor()

    for i, r in df.iterrows():
        s3_key = str(r["s3_key"])
        path = Path(local_solver_dir) / s3_key
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue

        ex = x.extract(
            str(path),
            ctx=str(r["ctx"]),
            ip_pos=str(r["ip_actor_flop"] or str(r["positions"]).split("v")[0]),
            oop_pos=str(r["oop_actor_flop"] or str(r["positions"]).split("v")[1]),
            board=str(r["board"] or ""),
            pot_bb=float(r["pot_bb"] or 0.0),
            stack_bb=float(r["effective_stack_bb"] or 0.0),
            bet_sizing_id=str(r.get("bet_sizing_id","")),
        )

        print(f"\n{i+1}/{len(df)}  {path.name}")
        if ex.ok:
            print("  OK  | root:", _top_k(ex.root_mix), "facing:", _top_k(ex.facing_mix),
                  "facing_bet_bb:", ex.facing_bet_bb, "via:", ex.meta.get("facing_path"))
        else:
            print("  BAD | reason:", ex.reason, "meta:", ex.meta)

def _top_k(d, k=3):
    if not d: return []
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [f"{k}:{v:.3f}" for k,v in items]

if __name__ == "__main__":
    # Example:
    # python tools/probe_solver_files.py data/artifacts/rangenet_postflop_flop_manifest.parquet /Users/you/S3 30
    manifest = sys.argv[1]
    local_dir = sys.argv[2]
    sample = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    main(manifest, local_dir, sample)
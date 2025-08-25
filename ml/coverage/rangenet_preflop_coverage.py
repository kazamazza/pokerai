import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def coverage_rangenet_preflop(parquet_path: str | Path) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    gcols = ["stack_bb", "hero_pos", "opener_pos", "opener_action"]

    grp = (df
           .groupby(gcols, as_index=False)
           .agg(n_rows=("weight","size"),
                total_weight=("weight","sum")))

    # simple thresholds – tweak as you like
    grp["ok"] = (grp["n_rows"] >= 1) & (grp["total_weight"] > 0)

    summary = {
        "rangenet_preflop": {
            "total_cells": int(len(grp)),
            "ok_cells": int(grp["ok"].sum()),
            "ok_pct": float(100.0 * grp["ok"].mean() if len(grp) else 0.0),
        }
    }
    return {"summary": summary, "cells": grp.to_dict(orient="records")}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="reports/coverage/rangenet_preflop.json")
    args = ap.parse_args()

    cov = coverage_rangenet_preflop(args.parquet)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cov, indent=2))
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()
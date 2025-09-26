import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import json

FALLBACK_COLS = [
    "range_source",
    "range_ip_fallback_level", "range_oop_fallback_level",
    "range_ip_stack_delta", "range_oop_stack_delta",
    "range_ip", "range_oop",
]

def _bool_series_or_false(s):
    if s is None:
        return pd.Series([False])
    return s.fillna(False).astype(bool)

def _contains_fallback(x: pd.Series) -> pd.Series:
    # string contains 'fallback', 'default', or empty/NaN ranges
    s = x.fillna("")
    s_low = s.astype(str).str.lower()
    return (
        s_low.str.contains("fallback")
        | s_low.str.contains("default")
        | (s.str.len() == 0)
    )

def _large_delta(ip_delta: pd.Series, oop_delta: pd.Series, max_abs: int) -> pd.Series:
    a = ip_delta.fillna(0).astype(float).abs() > max_abs
    b = oop_delta.fillna(0).astype(float).abs() > max_abs
    return a | b

def _coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)

def find_fallback_rows(df: pd.DataFrame, *, limp_only: bool, max_stack_delta: int) -> pd.DataFrame:
    need = [c for c in FALLBACK_COLS if c not in df.columns]
    if need:
        raise ValueError(f"Manifest missing required columns: {need}")

    # Any indicator of fallback / default / missing range
    is_fb = (
            _contains_fallback(df["range_source"])
            | (_coerce_numeric(df["range_ip_fallback_level"]) > 0)
            | (_coerce_numeric(df["range_oop_fallback_level"]) > 0)
            | (df["range_ip"].isna() | (df["range_ip"].astype(str).str.len() == 0))
            | (df["range_oop"].isna() | (df["range_oop"].astype(str).str.len() == 0))
            | _large_delta(
        _coerce_numeric(df["range_ip_stack_delta"]),
        _coerce_numeric(df["range_oop_stack_delta"]),
        max_stack_delta
    )
    )

    res = df[is_fb].copy()

    if limp_only:
        res = res[res["ctx"].astype(str).isin(["LIMPED_SINGLE", "LIMPED_MULTI"])]

    return res.reset_index(drop=True)

def dedupe_jobs(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the first instance of each sha1 (that’s the unit of work for your solver)
    if "sha1" not in df.columns:
        raise ValueError("Manifest lacks 'sha1' column; cannot dedupe jobs.")
    return df.drop_duplicates(subset=["sha1"]).reset_index(drop=True)

def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

def _to_jsonable(v: Any) -> Any:
    # scalars
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    # arrays
    if isinstance(v, np.ndarray):
        return [_to_jsonable(x) for x in v.tolist()]
    # pandas NA
    try:
        if pd.isna(v):  # catches np.nan, pd.NA
            return None
    except Exception:
        pass
    # containers
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_to_jsonable(x) for x in v)
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    return v

def write_jobs_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write minimal JSONL job envelopes for your existing solver queue.
    Keeps s3_key and sha1; converts NumPy types to JSON-safe Python.
    """
    cols_present = set(df.columns)
    pass_through = [
        "sha1", "s3_key",
        "street", "ctx", "topology",
        "positions", "ip_actor_flop", "oop_actor_flop",
        "opener", "three_bettor",
        "board_cluster_id", "board",
        "bet_sizing_id", "bet_sizes",
        "effective_stack_bb", "pot_bb",
        "rake_tier",
        "range_ip", "range_oop",
        "solver_version", "accuracy", "max_iter", "allin_threshold",
    ]
    use_cols = [c for c in pass_through if c in cols_present]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # iterate rows as dicts to avoid dtype surprises
        for rec in df[use_cols].to_dict(orient="records"):
            job: Dict[str, Any] = {k: _to_jsonable(rec.get(k)) for k in use_cols}
            job["force_resolve"] = True
            job["repair_reason"] = "fallback_range_detected"
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

def summarize(df_all: pd.DataFrame, df_bad: pd.DataFrame, df_jobs: pd.DataFrame) -> None:
    print(f"total rows in manifest: {len(df_all):,}")
    print(f"rows flagged as fallback: {len(df_bad):,}")
    print(f"unique jobs to (re)solve: {len(df_jobs):,}")
    if not df_jobs.empty:
        by_ctx = df_jobs.groupby("ctx").size().sort_values(ascending=False)
        print("\nby ctx:")
        print(by_ctx.to_string())
        if "positions" in df_jobs.columns:
            by_pos = df_jobs.groupby("positions").size().sort_values(ascending=False).head(20)
            print("\nby positions (top 20):")
            print(by_pos.to_string())

def main():
    ap = argparse.ArgumentParser(description="Find fallback/default range rows in flop manifest and emit repair jobs.")
    ap.add_argument("--manifest", type=str, default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--out-parquet", type=str, default="data/artifacts/rangenet_postflop_repair_manifest.parquet")
    ap.add_argument("--out-jobs", type=str, default="data/artifacts/rangenet_postflop_repair_jobs.jsonl")
    ap.add_argument("--limp-only", action="store_true", help="only repair limped contexts (LIMPED_SINGLE/MULTI)")
    ap.add_argument("--max-stack-delta", type=int, default=250, help="flag rows with |stack_delta| > this as fallback")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")

    df = pd.read_parquet(manifest)
    df_bad = find_fallback_rows(df, limp_only=args.limp_only, max_stack_delta=args.max_stack_delta)
    df_jobs = dedupe_jobs(df_bad)

    write_parquet(df_jobs, Path(args.out_parquet))
    write_jobs_jsonl(df_jobs, Path(args.out_jobs))
    summarize(df, df_bad, df_jobs)

    print(f"\n✅ wrote repair manifest → {args.out_parquet}")
    print(f"✅ wrote repair jobs     → {args.out_jobs}")

if __name__ == "__main__":
    main()
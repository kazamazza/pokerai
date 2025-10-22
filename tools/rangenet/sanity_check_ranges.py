import argparse
import sys
from pathlib import Path
import pandas as pd

DEFAULT_MANIFEST = Path("data/artifacts/rangenet_postflop_flop_manifest_NL10.parquet")

POS_COL_CANDIDATES = [
    ("ip_pos", "oop_pos"),
    ("ip_actor_flop", "oop_actor_flop"),  # often used in your sanity output
    ("ip", "oop"),
]

STACK_COL_CANDIDATES = ["stack_bb", "effective_stack_bb", "stack"]

CTX_COL_CANDIDATES = ["ctx", "context"]

def _choose_cols(df: pd.DataFrame, candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    raise KeyError(
        f"Could not locate position columns. Tried: {candidates}. "
        f"Available: {list(df.columns)}"
    )

def _choose_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not locate required column. Tried: {candidates}. "
        f"Available: {list(df.columns)}"
    )

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="List missing SPH pairs (range_source == 'fallback:default').")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Input manifest parquet path.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/artifacts"), help="Directory for outputs.")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 2

    df = pd.read_parquet(args.manifest)
    cols_lower = {c.lower(): c for c in df.columns}

    if "range_source" not in df.columns:
        print("[ERROR] Manifest lacks 'range_source' column; cannot detect fallbacks.")
        return 2

    # Normalize + filter for fallback:default
    df["range_source_norm"] = (
        df["range_source"].astype(str).str.strip().str.lower()
    )
    missing = df[df["range_source_norm"] == "fallback:default"].copy()

    if missing.empty:
        print("✅ No fallback:default rows found. Nothing to do.")
        return 0

    # Column selection with fallbacks
    ip_col, oop_col = _choose_cols(missing, POS_COL_CANDIDATES)
    stack_col = _choose_col(missing, STACK_COL_CANDIDATES)
    ctx_col = _choose_col(missing, CTX_COL_CANDIDATES)
    topology_col = "topology" if "topology" in missing.columns else None

    # Canonical view
    select_cols = [ctx_col, stack_col, ip_col, oop_col, "range_source"]
    if topology_col:
        select_cols.append(topology_col)

    out = (
        missing[select_cols]
        .rename(columns={
            ctx_col: "ctx",
            stack_col: "stack_bb",
            ip_col: "ip_pos",
            oop_col: "oop_pos",
        })
    )

    # Clean + dedupe
    out["ctx"] = out["ctx"].astype(str).str.upper().str.strip()
    out["ip_pos"] = out["ip_pos"].astype(str).str.upper().str.strip()
    out["oop_pos"] = out["oop_pos"].astype(str).str.upper().str.strip()
    out["stack_bb"] = pd.to_numeric(out["stack_bb"], errors="coerce").astype("Int64")
    out = (
        out.dropna(subset=["stack_bb"])
           .drop_duplicates(subset=["ctx", "stack_bb", "ip_pos", "oop_pos"])
           .sort_values(["ctx", "stack_bb", "ip_pos", "oop_pos"])
           .reset_index(drop=True)
    )

    # Summary
    summary = out.groupby(["ctx"])["oop_pos"].count().sort_values(ascending=False)
    print("📊 Missing (fallback:default) by ctx:")
    for ctx, n in summary.items():
        print(f"  - {ctx}: {n} pairs")

    # Output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "missing_sph_pairs.csv"
    pq_path = args.out_dir / "missing_sph_pairs.parquet"
    out.to_csv(csv_path, index=False)
    out.to_parquet(pq_path, index=False)

    print(f"✅ Wrote: {csv_path}  (rows={len(out)})")
    print(f"✅ Wrote: {pq_path}  (rows={len(out)})")
    print(out.head(12).to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
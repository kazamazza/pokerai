#!/usr/bin/env python3
# save as: tools/rangenet/sanity/make_flop_manifest_smoke.py

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser("Build a small, stratified postflop manifest for smoke runs")
    ap.add_argument("--in",  dest="in_path",  required=True,
                    help="Input parquet (e.g. data/artifacts/rangenet_postflop_flop_manifest_NL10.parquet)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output parquet for smoke runs")
    ap.add_argument("--per-ctx", type=int, default=5,
                    help="Rows to sample per ctx (default 5)")
    ap.add_argument("--contexts", type=str, default=None,
                    help="Comma-separated ctx filter (e.g. VS_OPEN,BLIND_VS_STEAL). Defaults to all found.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--dedupe-keys", type=str, default="ctx,positions,effective_stack_bb,board,bet_sizing_id",
                    help="Comma-separated columns to drop duplicates on before sampling")
    args = ap.parse_args()

    in_p  = Path(args.in_path)
    out_p = Path(args.out_path)
    if not in_p.exists():
        raise FileNotFoundError(in_p)

    df = pd.read_parquet(in_p)

    # Optional ctx filtering
    if args.contexts:
        keep = {c.strip().upper() for c in args.contexts.split(",") if c.strip()}
        # normalize a bit
        df["ctx"] = df["ctx"].astype(str).str.upper()
        before = len(df)
        df = df[df["ctx"].isin(keep)].reset_index(drop=True)
        print(f"ctx filter: kept {len(df)}/{before} rows for {sorted(keep)}")

    if len(df) == 0:
        raise SystemExit("No rows after filtering.")

    # Optional de-duplication on a stable key set (helps avoid near-duplicates)
    dedupe_cols = [c.strip() for c in args.dedupe_keys.split(",") if c.strip()]
    missing = [c for c in dedupe_cols if c not in df.columns]
    if missing:
        print(f"[warn] dedupe keys missing in manifest: {missing} (skipping those)")
        dedupe_cols = [c for c in dedupe_cols if c in df.columns]

    if dedupe_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
        print(f"dedupe: {before} → {len(df)} rows using keys {dedupe_cols}")

    # Group by ctx and sample up to N per group (stable random seed)
    if "ctx" not in df.columns:
        raise SystemExit("Manifest missing 'ctx' column; cannot stratify.")

    per = max(1, int(args.per_ctx))
    parts = []
    counts = []
    for ctx_val, g in df.groupby("ctx", sort=True):
        take = min(per, len(g))
        sample = g.sample(n=take, random_state=args.seed) if take < len(g) else g
        parts.append(sample)
        counts.append((ctx_val, take, len(g)))

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # optional final shuffle

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_p, index=False)
    print(f"✅ wrote {len(out)} rows → {out_p}")
    print("Breakdown (ctx: picked / available):")
    for ctx_val, picked, avail in counts:
        print(f" - {ctx_val}: {picked} / {avail}")

if __name__ == "__main__":
    main()
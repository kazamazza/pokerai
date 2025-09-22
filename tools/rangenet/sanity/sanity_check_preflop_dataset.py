#!/usr/bin/env python3
# sanity_check_preflop_dataset.py
import argparse, math, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

def find_y_cols(df: pd.DataFrame) -> list[str]:
    ys = [c for c in df.columns if c.startswith("y_")]
    ys = sorted(ys, key=lambda s: int(s.split("_",1)[1]))
    if len(ys) != 169:
        raise SystemExit(f"Expected 169 y_* columns, found {len(ys)}.")
    return ys

def load_all_hands() -> list[str]:
    """
    Try to import your canonical 169-hand ordering. If not found,
    fall back to y_0..y_168 indices.
    """
    try:
        # adjust to your project if you have a canonical list
        from ml.poker.hands169 import ALL_HANDS  # noqa
        if len(ALL_HANDS) == 169:
            return list(ALL_HANDS)
    except Exception:
        pass
    return [f"#{i}" for i in range(169)]

def entropy(p: np.ndarray) -> float:
    q = np.clip(p, 1e-12, 1.0)
    return float(-(q * np.log(q)).sum())

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def main():
    ap = argparse.ArgumentParser("Sanity-check a preflop RangeNet parquet")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sample", type=int, default=8, help="rows to print detailed tops for")
    ap.add_argument("--uniform-tol", type=float, default=1e-6, help="max deviation vs 1/169 to flag uniform")
    ap.add_argument("--report-csv", type=str, default=None, help="optional path to save per-row metrics")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    y_cols = find_y_cols(df)
    H = load_all_hands()
    uni_val = 1.0/169.0

    # --- per-row metrics ---
    Y = df[y_cols].to_numpy(dtype=np.float64)
    sums = Y.sum(axis=1)
    mins = Y.min(axis=1)
    maxs = Y.max(axis=1)
    nnz  = (Y > 1e-12).sum(axis=1)
    ents = np.array([entropy(Y[i]) for i in range(len(df))])
    is_nan = np.isnan(Y).any(axis=1)
    is_uniform = (np.abs(Y - uni_val).max(axis=1) <= args.uniform_tol)

    # pairwise similarity (coarse)
    cos_warn = None
    if len(df) >= 3:
        sims = []
        for i in range(min(len(df), 100)):
            for j in range(i+1, min(len(df), 100)):
                sims.append(cosine(Y[i], Y[j]))
        if sims:
            sims = np.array(sims, dtype=float)
            cos_warn = (float(np.median(sims)), float(np.percentile(sims, 95)))

    # --- coverage snapshot ---
    by_ctx = df.groupby("ctx").size().to_dict() if "ctx" in df else {}
    by_stack = df.groupby("stack_bb").size().to_dict() if "stack_bb" in df else {}
    by_pair = df.groupby(["opener_pos","hero_pos"]).size().to_dict() if {"opener_pos","hero_pos"} <= set(df.columns) else {}

    # --- print summary ---
    print(f"📦 File: {args.parquet}")
    print(f"rows: {len(df)}  | y-cols: {len(y_cols)}")
    print(f"NaN rows: {int(is_nan.sum())}")
    print(f"non-normalized rows (|sum-1|>1e-6): {int(np.sum(np.abs(sums-1.0) > 1e-6))}")
    print(f"uniform rows (≈1/169): {int(is_uniform.sum())} / {len(df)}")
    print(f"min(y)∈[{mins.min():.4g},{mins.max():.4g}]  max(y)∈[{maxs.min():.4g},{maxs.max():.4g}]")
    print(f"nnz mean={nnz.mean():.1f}  min={nnz.min()}  max={nnz.max()}")
    print(f"entropy nat mean={ents.mean():.3f}  min={ents.min():.3f}  max={ents.max():.3f}  (uniform=ln(169)≈{math.log(169):.3f})")
    if cos_warn:
        med, p95 = cos_warn
        print(f"cosine similarity median={med:.4f}, p95={p95:.4f}  (warn if ≳0.995 everywhere)")

    if by_ctx:
        print("by ctx:", "  ".join(f"{k}:{v}" for k,v in by_ctx.items()))
    if by_stack:
        print("by stack:", "  ".join(f"{k}:{v}" for k,v in sorted(by_stack.items())))
    if by_pair:
        top_pairs = sorted(by_pair.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        print("top opener/hero pairs:", ", ".join([f"{a}v{b}:{n}" for ((a,b),n) in top_pairs]))

    # --- sample a few rows and show top hands ---
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(df), size=min(args.sample, len(df)), replace=False)
    print("\n🔎 Samples (top-10 hands):")
    for i in idxs:
        row = df.iloc[i]
        p = Y[i]
        top_idx = np.argsort(p)[::-1][:10]
        tops = [f"{H[j]}:{p[j]:.3f}" for j in top_idx]
        key = {k: row[k] for k in ["ctx","stack_bb","opener_pos","hero_pos"] if k in df.columns}
        tag = "UNIFORM?" if np.allclose(p, uni_val, atol=args.uniform_tol) else ""
        print(f"  [{i}] {key}  sum={sums[i]:.6f}  H={ents[i]:.3f}  {tag}")
        print("      " + ", ".join(tops))

    # optional CSV
    if args.report_csv:
        out = df.copy()
        out["sum"] = sums; out["nnz"] = nnz; out["entropy_nat"] = ents; out["is_uniform"] = is_uniform
        out.to_csv(args.report_csv, index=False)
        print(f"\n📝 Wrote per-row report → {args.report_csv}")

if __name__ == "__main__":
    main()
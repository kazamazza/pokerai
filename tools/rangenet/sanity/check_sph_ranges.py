#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


RANKS = "AKQJT98765432"

def _hand_to_index(code: str) -> int:
    """
    Map compact hand code to 169 index:
      - pairs: 'AA','KK',... (row==col)
      - suited: 'AKs' → row=A col=K (upper triangle)
      - offsuit: 'AKo' → row=K col=A (lower triangle)
    Row-major: row * 13 + col, rows/cols follow RANKS order.
    """
    code = code.strip()
    if len(code) == 2:
        r = RANKS.index(code[0])
        return r * 13 + r
    if len(code) == 3:
        r1, r2, s = code[0], code[1], code[2]
        i = RANKS.index(r1)
        j = RANKS.index(r2)
        if s == "s":
            return i * 13 + j
        elif s == "o":
            return j * 13 + i
    raise ValueError(f"Bad hand code: {code}")


def _iter_pairs(cache_root: Path):
    """Yield (ctx, stack:int, pair:str, pair_dir:Path) under cache_root."""
    for ctx_dir in sorted(cache_root.iterdir()):
        if not ctx_dir.is_dir():
            continue
        ctx = ctx_dir.name.upper()
        if ctx not in {"SRP", "LIMP_SINGLE", "LIMP_MULTI"}:
            continue
        for stack_dir in sorted(ctx_dir.iterdir()):
            if not stack_dir.is_dir():
                continue
            try:
                stack = int(stack_dir.name)
            except Exception:
                continue
            for pair_dir in sorted(stack_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                yield ctx, stack, pair_dir.name, pair_dir

def _load_monker_file(p: Path) -> Optional[np.ndarray]:
    if not p.exists():
        return None
    s = p.read_text(encoding="utf-8").strip()
    return monker_string_to_vec169(s)

def check_pair(ctx: str, stack: int, pair: str, pair_dir: Path) -> Dict:
    ip_path  = pair_dir / "ip.csv"
    oop_path = pair_dir / "oop.csv"

    ip_vec  = _load_monker_file(ip_path)
    oop_vec = _load_monker_file(oop_path)

    row = {
        "ctx": ctx,
        "stack": stack,
        "pair": pair,
        "ip_path": str(ip_path),
        "oop_path": str(oop_path),
        "ip_exists": ip_vec is not None,
        "oop_exists": oop_vec is not None,
        "ip_nnz": None,
        "oop_nnz": None,
        "ip_sum": None,
        "oop_sum": None,
        "ip_min": None,
        "ip_max": None,
        "oop_min": None,
        "oop_max": None,
        "ip_ok": False,
        "oop_ok": False,
        "issues": [],
    }

    # IP
    if ip_vec is None:
        row["issues"].append("missing_ip")
    else:
        row["ip_nnz"] = int((ip_vec > 0).sum())
        row["ip_sum"] = float(ip_vec.sum())
        row["ip_min"] = float(ip_vec.min()) if ip_vec.size else None
        row["ip_max"] = float(ip_vec.max()) if ip_vec.size else None
        if not np.all(np.isfinite(ip_vec)):
            row["issues"].append("ip_nan")
        if row["ip_min"] is not None and row["ip_min"] < 0.0:
            row["issues"].append("ip_below_zero")
        if row["ip_max"] is not None and row["ip_max"] > 1.0:
            row["issues"].append("ip_above_one")
        # heuristic sanity: nnz should not be extremely tiny
        if row["ip_nnz"] is not None and row["ip_nnz"] < 5:
            row["issues"].append("ip_too_sparse")
        row["ip_ok"] = ("missing_ip" not in row["issues"]
                        and "ip_nan" not in row["issues"]
                        and "ip_below_zero" not in row["issues"]
                        and "ip_above_one" not in row["issues"])

    # OOP
    if oop_vec is None:
        row["issues"].append("missing_oop")
    else:
        row["oop_nnz"] = int((oop_vec > 0).sum())
        row["oop_sum"] = float(oop_vec.sum())
        row["oop_min"] = float(oop_vec.min()) if oop_vec.size else None
        row["oop_max"] = float(oop_vec.max()) if oop_vec.size else None
        if not np.all(np.isfinite(oop_vec)):
            row["issues"].append("oop_nan")
        if row["oop_min"] is not None and row["oop_min"] < 0.0:
            row["issues"].append("oop_below_zero")
        if row["oop_max"] is not None and row["oop_max"] > 1.0:
            row["issues"].append("oop_above_one")
        # heuristic sanity: defend should not be all-zero
        if row["oop_nnz"] is not None and row["oop_nnz"] < 5:
            row["issues"].append("oop_too_sparse")
        row["oop_ok"] = ("missing_oop" not in row["issues"]
                         and "oop_nan" not in row["issues"]
                         and "oop_below_zero" not in row["issues"]
                         and "oop_above_one" not in row["issues"])

    return row

def main():
    ap = argparse.ArgumentParser(description="Sanity check packed SPH ranges (ip.csv/oop.csv as Monker strings).")
    ap.add_argument("--cache-root", type=Path, default=Path("data/vendor_cache/sph"),
                    help="Root directory with SRP/LIMP_SINGLE/LIMP_MULTI subdirs.")
    ap.add_argument("--out", type=Path, default=None, help="Optional CSV report to write.")
    ap.add_argument("--show-bad", action="store_true", help="Print problematic pairs.")
    args = ap.parse_args()

    if not args.cache_root.exists():
        print(f"❌ cache_root not found: {args.cache_root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for ctx, stack, pair, pair_dir in _iter_pairs(args.cache_root):
        rows.append(check_pair(ctx, stack, pair, pair_dir))

    if not rows:
        print("No pairs found. Expected structure: data/vendor_cache/sph/<CTX>/<STACK>/<IP_OOP>/ip.csv|oop.csv")
        sys.exit(2)

    df = pd.DataFrame(rows)
    df["ok"] = df["ip_ok"] & df["oop_ok"]
    total = len(df)
    ok = int(df["ok"].sum())
    bad = total - ok

    # Summary
    print(f"Checked {total} pairs under {args.cache_root}")
    print(df.groupby(["ctx", "stack"])["ok"].agg(["count", "sum"]).rename(columns={"count": "pairs", "sum": "ok"}).to_string())

    if bad > 0 and args.show_bad:
        bad_df = df[~df["ok"]].copy()
        print("\n⚠️  Problematic pairs:")
        cols = ["ctx","stack","pair","issues","ip_nnz","oop_nnz","ip_sum","oop_sum","ip_min","ip_max","oop_min","oop_max"]
        print(bad_df[cols].to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n💾 wrote report → {args.out}")

    # Exit non-zero if anything failed (CI-friendly)
    sys.exit(0 if bad == 0 else 3)

if __name__ == "__main__":
    main()
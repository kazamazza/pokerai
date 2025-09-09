#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd



ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
# --- use your existing utils (no duplication) ---
# adjust the import path to where you keep these
from ml.etl.utils.range_lookup import monker_string_to_vec169
from ml.etl.utils.monker_range_converter import to_monker


def _vec_stats_from_monker(s: str) -> tuple[int, float, bool]:
    """
    Return (nnz, sum, ok_in_01) from a Monker string; (-1,-1,False) if parse fails.
    """
    try:
        v = monker_string_to_vec169(s)
        a = np.asarray(v, dtype=float)
        if a.size != 169:
            return -1, -1.0, False
        ok_01 = bool(np.all(np.isfinite(a)) and np.min(a) >= 0.0 and np.max(a) <= 1.0)
        nnz = int((a > 0).sum())
        return nnz, float(a.sum()), ok_01
    except Exception:
        return -1, -1.0, False


def _to_monker_guard(payload) -> str:
    """
    Normalize whatever the manifest has (JSON-169, 13x13, already Monker, etc.)
    to a Monker CSV string using your existing helper.
    """
    # payload can be str(JSON) or list; to_monker handles both
    return to_monker(payload)


def main():
    ap = argparse.ArgumentParser(description="Sanity-check all ranges in a RangeNet flop manifest parquet.")
    ap.add_argument("--manifest", type=Path, default=Path("data/artifacts/rangenet_postflop_flop_manifest.parquet"))
    ap.add_argument("--min-nnz", type=int, default=10, help="Minimum nonzero entries required per range")
    ap.add_argument("--sample", type=int, default=8, help="How many bad rows to print")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_parquet(args.manifest)
    need = {"sha1", "range_ip", "range_oop"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"Manifest missing required columns: {miss}", file=sys.stderr)
        sys.exit(2)

    bad_rows = []
    total = len(df)
    ok = 0

    # helpful columns to display if present
    show_cols = [c for c in (
        "sha1", "positions", "effective_stack_bb", "pot_bb", "board", "bet_sizing_id",
        "range_ip_source_path", "range_oop_source_path",
        "range_ip_source_stack", "range_oop_source_stack",
        "range_ip_fallback_level", "range_oop_fallback_level",
        "range_pair_substituted", "ctx"
    ) if c in df.columns]

    for idx, row in df.iterrows():
        try:
            # Normalize to Monker strings
            s_ip = _to_monker_guard(row["range_ip"])
            s_oop = _to_monker_guard(row["range_oop"])

            # Back to vectors for numeric checks
            ip_nnz, ip_sum, ip_ok = _vec_stats_from_monker(s_ip)
            oop_nnz, oop_sum, oop_ok = _vec_stats_from_monker(s_oop)

            ip_good = (ip_ok and ip_nnz >= args.min_nnz)
            oop_good = (oop_ok and oop_nnz >= args.min_nnz)

            if ip_good and oop_good:
                ok += 1
            else:
                bad_rows.append({
                    **{k: row.get(k, None) for k in show_cols},
                    "ip_nnz": ip_nnz, "ip_sum": round(ip_sum, 2), "ip_ok01": ip_ok,
                    "oop_nnz": oop_nnz, "oop_sum": round(oop_sum, 2), "oop_ok01": oop_ok,
                })

        except Exception as e:
            bad_rows.append({
                **{k: row.get(k, None) for k in show_cols},
                "error": str(e),
            })

    n_bad = len(bad_rows)
    print(f"Checked {total} manifest rows")
    print(f"  ✅ ok: {ok}")
    print(f"  ❌ bad: {n_bad}  (min_nnz={args.min_nnz})")

    if n_bad:
        print("\nExamples of bad rows:")
        sample = bad_rows[:args.sample]
        # order columns nicely
        cols = [c for c in (
            "sha1", "ctx", "positions", "effective_stack_bb", "pot_bb", "board", "bet_sizing_id",
            "ip_nnz", "ip_sum", "ip_ok01", "oop_nnz", "oop_sum", "oop_ok01",
            "range_pair_substituted",
            "range_ip_source_stack", "range_oop_source_stack",
            "range_ip_source_path", "range_oop_source_path",
            "error"
        ) if c in sample[0].keys()]
        sdf = pd.DataFrame(sample)[cols]
        print(sdf.to_string(index=False))

        # exit nonzero so CI/scripts can fail-fast
        sys.exit(1)
    else:
        print("\n✅ Preflight OK: all rows have healthy, normalizable ranges.")
        sys.exit(0)


if __name__ == "__main__":
    main()
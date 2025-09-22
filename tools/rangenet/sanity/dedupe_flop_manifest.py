import argparse
import pandas as pd
import numpy as np

CRITICAL_EQUAL = [
    "street","positions","board","board_cluster_id",
    "effective_stack_bb","pot_bb","bet_sizing_id","node_key",
]

PREF_NUMERIC_MAX = ["max_iter"]          # prefer higher if differs
PREF_NUMERIC_MIN = ["accuracy"]          # prefer lower if differs
PREF_EXISTS_MORE = [                     # prefer row with more filled values
    "range_ip","range_oop","solver_version","parent_sha1","parent_node_key","line_key"
]

def _score_row(r):
    score = 0.0
    # more filled metadata is better
    score += sum(int(pd.notna(r.get(c))) for c in PREF_EXISTS_MORE)
    # prefer tighter accuracy (smaller)
    for c in PREF_NUMERIC_MIN:
        v = r.get(c)
        if pd.notna(v): score += 0.1 / max(1e-9, float(v))
    # prefer larger max_iter
    for c in PREF_NUMERIC_MAX:
        v = r.get(c)
        if pd.notna(v): score += 0.0001 * float(v)
    return score

def main():
    ap = argparse.ArgumentParser("Deduplicate flop manifest by s3_key")
    ap.add_argument("in_parquet")
    ap.add_argument("-o","--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    n0 = len(df)

    if "s3_key" not in df.columns:
        raise SystemExit("manifest missing column: s3_key")

    # Optional: quick sanity
    if "sha1" not in df.columns:
        df["sha1"] = np.nan

    # Group by s3_key and pick a canonical row
    picks = []
    conflicts = 0
    for key, g in df.groupby("s3_key", sort=False):
        if len(g) == 1:
            picks.append(g.index[0])
            continue

        # Check if critical fields conflict; if yes, just pick the best-scored row but log it
        same = True
        for c in CRITICAL_EQUAL:
            if c in g.columns:
                vals = g[c].astype(str).fillna("<NA>").unique()
                if len(vals) > 1:
                    same = False
                    break

        if not same:
            conflicts += 1

        # pick row with highest heuristic score
        scores = g.apply(_score_row, axis=1)
        best_idx = scores.idxmax()
        picks.append(best_idx)

    dedup = df.loc[picks].copy().reset_index(drop=True)

    n1 = len(dedup)
    n_dupes = n0 - n1
    print(f"rows: {n0}  unique s3_key: {n1}  dupes removed: {n_dupes}")
    if conflicts:
        print(f"⚠️ {conflicts} s3_key groups had conflicting critical fields; chose best row per score.")

    out = args.out or args.in_parquet.replace(".parquet", ".dedup.parquet")
    dedup.to_parquet(out, index=False)
    print(f"✅ wrote {out}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
import argparse
from typing import Counter
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/artifacts/monker_manifest.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    pairs = df[["opener_pos", "hero_pos"]].dropna().drop_duplicates()
    print("✅ Unique opener v hero pairs:")
    for op, hero in sorted(pairs.itertuples(index=False, name=None)):
        print(f"  {op} v {hero}")

    # optional counts
    print("\nCounts:")
    cnt = Counter(map(tuple, df[["opener_pos", "hero_pos"]].dropna().itertuples(index=False, name=None)))
    for (op, hero), n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        print(f"{op} v {hero}: {n}")

if __name__ == "__main__":
    main()
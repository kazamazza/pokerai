#!/usr/bin/env python3
import argparse, json, re
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.monker_range_converter import to_monker, monker_to_vec169



def main():
    ap = argparse.ArgumentParser(description="Sanity check: normalize a file to Monker string and preview.")
    ap.add_argument("--file", required=True, help="Path to SPH ip.csv/oop.csv or raw text")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8").strip()

    monker_str = to_monker(text)
    vec = monker_to_vec169(monker_str)

    nnz = int(np.count_nonzero(vec))
    s = float(vec.sum())

    print(f"[post] nnz={nnz} sum={s:.2f}")
    print(f"[monker] {monker_str[:180]}{'…' if len(monker_str) > 180 else ''}")


if __name__ == "__main__":
    main()
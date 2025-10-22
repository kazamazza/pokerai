# tools/rangenet/sanity/sanity_check_ip_oop_duplicates.py
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

EPS = 1e-6

def _as_path_if_exists(s: Any) -> Optional[Path]:
    if isinstance(s, Path):
        return s if s.exists() else None
    if isinstance(s, str) and ("/" in s or s.endswith(".csv") or s.endswith(".txt") or s.endswith(".parquet")):
        p = Path(s)
        return p if p.exists() else None
    return None

def _parse_grid_13x13_from_text(txt: str) -> np.ndarray:
    # Accept: 13 newline rows of comma/space-separated floats
    lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    # JSON array?
    if txt.strip().startswith("["):
        arr = json.loads(txt)
        a = np.array(arr, dtype=float)
        if a.size == 169:
            return a.reshape(169)
        raise ValueError(f"JSON size {a.shape}, expected 169")
    # CSV-like (13 rows)
    if len(lines) == 13:
        rows = []
        for ln in lines:
            parts = re.split(r"[,\s]+", ln.strip())
            parts = [p for p in parts if p != ""]
            rows.append([float(x) for x in parts])
        a = np.array(rows, dtype=float)
        if a.shape != (13,13):
            raise ValueError(f"Parsed shape {a.shape}, expected 13x13")
        return a.reshape(169)
    # Flat sequence of 169 numbers
    parts = re.split(r"[,\s]+", txt.strip())
    parts = [p for p in parts if p != ""]
    if len(parts) == 169:
        return np.array([float(x) for x in parts], dtype=float)
    raise ValueError("Unrecognized inline grid format")

def _load_grid_csv(path: Path) -> np.ndarray:
    arr = np.array(pd.read_csv(path, header=None), dtype=float)
    if arr.shape == (13,13):
        return arr.reshape(169)
    if arr.size == 169:
        return arr.reshape(169)
    raise ValueError(f"{path}: expected 13x13 or 169 numbers, got {arr.shape}")

def _load_range_cell(cell: Any) -> np.ndarray:
    """
    Load range vector (169) from a manifest cell:
    - If it's a path string and exists → read CSV.
    - If it's bytes → decode to text and parse.
    - If it's a list/ndarray → coerce.
    - Else assume inline text (CSV/JSON/flat) and parse.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        raise ValueError("empty cell")
    # existing file?
    p = _as_path_if_exists(cell)
    if p is not None:
        return _load_grid_csv(p)
    # already array-like?
    if isinstance(cell, (list, tuple, np.ndarray)):
        a = np.array(cell, dtype=float)
        if a.size == 169:
            return a.reshape(169)
        # maybe 13x13
        if a.shape == (13,13):
            return a.reshape(169)
        raise ValueError(f"array-like with size {a.size}, expected 169")
    # bytes?
    if isinstance(cell, (bytes, bytearray)):
        txt = cell.decode("utf-8", errors="ignore")
        return _parse_grid_13x13_from_text(txt)
    # string inline
    if isinstance(cell, str):
        return _parse_grid_13x13_from_text(cell)
    # fallback
    raise ValueError(f"Unsupported cell type {type(cell)}")

def _check_vec(name: str, v: np.ndarray, strict_binary: bool=False) -> List[str]:
    errs: List[str] = []
    if v.size != 169:
        errs.append(f"{name}: size {v.size}, expected 169")
        return errs
    if not np.all(np.isfinite(v)):
        errs.append(f"{name}: non-finite values present")
    if np.any(v < -EPS) or np.any(v > 1.0 + EPS):
        errs.append(f"{name}: values out of [0,1] (min={np.min(v):.4f}, max={np.max(v):.4f})")
    # duplicates in hand-list path show up as weights >1 once aggregated
    if np.any(v > 1.0 + EPS):
        idxs = np.where(v > 1.0 + EPS)[0][:10]
        errs.append(f"{name}: duplicate labels detected at indices={idxs.tolist()}")
    if strict_binary and np.any((v > EPS) & (v < 1.0 - EPS)):
        errs.append(f"{name}: non-binary weights present")
    return errs

def _pick_id(dfrow) -> str:
    parts = []
    for c in ("scenario","ctx","positions","ip_actor_flop","oop_actor_flop","effective_stack_bb","board_cluster_id"):
        if hasattr(dfrow, c):
            parts.append(f"{c}={getattr(dfrow,c)}")
    return " ".join(parts) if parts else f"row={dfrow.Index}"

def main():
    ap = argparse.ArgumentParser(description="Sanity-check inline IP/OOP ranges for duplicates and bounds.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--strict-binary", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)

    # Column detection for inline ranges
    if not {"range_ip","range_oop"}.issubset(df.columns):
        print(f"[ERROR] Manifest lacks 'range_ip'/'range_oop'. Found columns: {list(df.columns)}")
        sys.exit(2)

    problems = 0
    checked = 0

    it = df.reset_index().itertuples()
    for row in it:
        if args.max_rows and checked >= args.max_rows:
            break
        row_errs: List[str] = []
        try:
            v_ip = _load_range_cell(getattr(row, "range_ip"))
            row_errs += _check_vec("IP", v_ip, args.strict_binary)
        except Exception as e:
            row_errs.append(f"IP load failed: {e}")
        try:
            v_oop = _load_range_cell(getattr(row, "range_oop"))
            row_errs += _check_vec("OOP", v_oop, args.strict_binary)
        except Exception as e:
            row_errs.append(f"OOP load failed: {e}")

        if row_errs:
            problems += 1
            print(f"❌ { _pick_id(row) }")
            for m in row_errs:
                print(f"   - {m}")
        else:
            checked += 1

    print(f"\nDone. checked={checked} problem_rows={problems}")
    sys.exit(1 if problems else 0)

if __name__ == "__main__":
    main()
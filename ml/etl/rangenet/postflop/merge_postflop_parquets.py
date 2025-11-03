import argparse, sys, json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import pandas as pd

ROOT_TOKENS   = ["CHECK","BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
                 "DONK_25","DONK_33","DONK_50","DONK_66","DONK_75","DONK_100"]
FACING_TOKENS = ["FOLD","CALL","RAISE_150","RAISE_200","RAISE_250","RAISE_300","RAISE_400","RAISE_500","ALLIN"]

META_KEEP = [
    "ctx","street","ip_pos","oop_pos","hero_pos","board","board_cluster_id",
    "pot_bb","eff_stack_bb","size_pct","size_frac","actor","stakes","stakes_id",
    "weight","valid","s3_key","bet_sizing_id","part_id"
]

def _find_parquets(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    return sorted(p for p in path.rglob("*.parquet"))

def _collect(paths: Iterable[str|Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        out += _find_parquets(Path(p))
    return out

def _ensure_columns(df: pd.DataFrame, cols: List[str], fill: float = 0.0, dtype="float32"):
    for c in cols:
        if c not in df.columns:
            df[c] = fill
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill).astype(dtype)

def _coerce_common(df: pd.DataFrame) -> pd.DataFrame:
    # light normalization of frequent fields
    for c in ("pot_bb","eff_stack_bb","size_frac","weight"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").fillna(0.0)
    if "size_pct" in df.columns:
        df["size_pct"] = pd.to_numeric(df["size_pct"], errors="coerce").astype("Int64")
    if "street" in df.columns:
        df["street"] = pd.to_numeric(df["street"], errors="coerce").astype("Int64")
    if "board_cluster_id" in df.columns:
        df["board_cluster_id"] = pd.to_numeric(df["board_cluster_id"], errors="coerce").fillna(0).astype("int32")
    # strings upper
    for c in ("ctx","ip_pos","oop_pos","hero_pos","actor","stakes","bet_sizing_id"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    return df

def _merge_kind(kind: str, inputs: List[Path], out_path: Path) -> Tuple[int, Dict[str,int]]:
    if not inputs:
        raise SystemExit(f"No {kind} parquet parts found.")
    frames = []
    for p in inputs:
        df = pd.read_parquet(p)
        df = _coerce_common(df)
        if kind == "root":
            _ensure_columns(df, ROOT_TOKENS, 0.0, "float32")
            # strip facing tokens if present
            drop_cols = [c for c in df.columns if any(c.startswith(t.split("_")[0]) for t in FACING_TOKENS)]
            df = df.drop(columns=drop_cols, errors="ignore")
        else:
            _ensure_columns(df, FACING_TOKENS, 0.0, "float32")
            # strip root tokens if present
            drop_cols = [c for c in df.columns if c=="CHECK" or c.startswith("BET_") or c.startswith("DONK_")]
            df = df.drop(columns=drop_cols, errors="ignore")
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)

    # final column order: meta → tokens
    tokens = ROOT_TOKENS if kind=="root" else FACING_TOKENS
    cols = [c for c in META_KEEP if c in merged.columns] + \
           sorted([c for c in merged.columns if c not in META_KEEP and c not in tokens]) + tokens
    merged = merged[[c for c in cols if c in merged.columns]]

    # quick sanity summaries
    if kind == "root":
        nz = merged[ROOT_TOKENS].gt(1e-9).sum().to_dict()
    else:
        nz = merged[FACING_TOKENS].gt(1e-9).sum().to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return len(merged), {k:int(v) for k,v in nz.items()}

def main():
    ap = argparse.ArgumentParser(description="Merge postflop parquet parts into final root/facing datasets")
    ap.add_argument("--root-parts", nargs="*", default=[], help="files/dirs containing ROOT parts")
    ap.add_argument("--facing-parts", nargs="*", default=[], help="files/dirs containing FACING parts")
    ap.add_argument("--out-root", required=True, help="output parquet file for ROOT")
    ap.add_argument("--out-facing", required=True, help="output parquet file for FACING")
    args = ap.parse_args()

    root_inputs   = _collect(args.root_parts)
    facing_inputs = _collect(args.facing_parts)

    n_root, nz_root   = _merge_kind("root", root_inputs, Path(args.out_root))
    n_face, nz_facing = _merge_kind("facing", facing_inputs, Path(args.out_facing))

    print("✅ Merge finished")
    print(f" • ROOT rows: {n_root}")
    print(f" • FACING rows: {n_face}")
    print(" • ROOT nonzero by token:", json.dumps(nz_root, indent=2))
    print(" • FACING nonzero by token:", json.dumps(nz_facing, indent=2))

if __name__ == "__main__":
    main()
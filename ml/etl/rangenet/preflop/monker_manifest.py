from __future__ import annotations
import argparse
import json
import hashlib
import sys
import tarfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client

# 🔽 Use unified helpers so everyone speaks the same taxonomy
from ml.etl.rangenet.preflop.monker_helpers import (
    POS_SET,  # {'UTG','HJ','CO','BTN','SB','BB'}
    canon_pos,  # alias normalizer (BU->BTN, EP->UTG, MP->HJ, etc.)
    parse_seq_from_stem,  # parses raw vendor tokens: [{"pos":"UTG","action":"Min"}, ...]
    first_non_fold_opener, canon_action,  # returns (pos, raw_action) where raw_action ∈ {'Min','AI',...}
)

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_stack_from_parts(parts: List[str]) -> int | None:
    for p in parts:
        q = p.lower()
        if q.endswith("bb"):
            try:
                return int(q[:-2])
            except ValueError:
                pass
    return None

def scan_monker(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in root.rglob("*.txt"):
        parts = list(path.parts)
        stack_bb = parse_stack_from_parts(parts)

        # hero folder is the immediate parent (vendor uses this as "hero POV")
        hero_pos = canon_pos(path.parent.name)
        if hero_pos not in POS_SET:
            hero_pos = None

        stem = path.stem
        # 🔑 parse raw vendor sequence (keep tokens like 'Min','AI','Call','Fold')
        seq_raw = parse_seq_from_stem(stem)  # raw vendor tokens
        opener_pos_raw, opener_action_raw = first_non_fold_opener(seq_raw)

        seq_canon = [
            {"pos": e["pos"], **({"action": canon_action(e.get("action"))} if "action" in e else {})}
            for e in seq_raw
        ]

        seq_json = json.dumps(seq_canon, sort_keys=True)  # stable representation
        file_sha1 = sha1_file(path)
        sig = sha1_str(f"{stack_bb}|{hero_pos}|{seq_json}")

        rows.append({
            "stack_bb": stack_bb,
            "hero_pos": hero_pos,
            # raw (for filename matching)
            "opener_pos_raw": opener_pos_raw,
            "opener_action_raw": opener_action_raw,  # can be Min/AI
            "sequence_raw": json.dumps(seq_raw, sort_keys=True),
            # canonical (for downstream logic)
            "opener_pos": opener_pos_raw,  # positions are already canonical
            "opener_action": canon_action(opener_action_raw),  # e.g., RAISE/ALL_IN
            "sequence": json.dumps(seq_canon, sort_keys=True),  # e.g., RAISE/ALL_IN/CALL/FOLD...
            "filename_stem": stem,
            "rel_path": str(path.relative_to(root)),
            "abs_path": str(path.resolve()),
            "file_sha1": file_sha1,
            "sig": sig,
        })

    df = pd.DataFrame(rows)

    # Keep first representative per (stack, hero, sequence) and count duplicates
    grouped = (
        df.groupby(["stack_bb", "hero_pos", "sequence"], dropna=False)
        .agg(
            opener_pos=("opener_pos", "first"),
            opener_action=("opener_action", "first"),
            # also carry raw if you want to debug:
            opener_pos_raw=("opener_pos_raw", "first"),
            opener_action_raw=("opener_action_raw", "first"),
            filename_stem=("filename_stem", "first"),
            rel_path=("rel_path", "first"),
            abs_path=("abs_path", "first"),
            file_sha1=("file_sha1", "first"),
            sig=("sig", "first"),
            n_files=("rel_path", "count"),
        )
        .reset_index()
    )
    return grouped

def write_manifest(df: pd.DataFrame, out_parquet: Path, out_jsonl: Path | None = None):
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_parquet, index=False)
        print(f"✅ wrote manifest → {out_parquet}")
    except Exception as e:
        print(f"⚠️ Parquet write failed ({e}); writing JSONL fallback.")
        if out_jsonl is None:
            out_jsonl = out_parquet.with_suffix(".jsonl")
        with out_jsonl.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec) + "\n")
        print(f"✅ wrote manifest (JSONL) → {out_jsonl}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/vendor/monker",
                    help="Root folder containing stack subfolders (e.g., 12bb, 15bb, ...)")
    ap.add_argument("--out", type=str, default="data/artifacts/monker_manifest.parquet")
    ap.add_argument("--s3-bucket", type=str, default="pokeraistore",
                    help="S3 bucket where monker.tar.gz is stored")
    ap.add_argument("--s3-key", type=str, default="data/vendor/monker.tar.gz",
                    help="S3 key for monker archive")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    # 1. Download archive from S3
    s3c = S3Client(bucket_name=args.s3_bucket)
    local_gz = Path("data/vendor/monker.tar.gz")
    s3c.download_file_if_missing(args.s3_key, local_gz)

    # 2. Extract archive if needed
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"▶️ Extracting {local_gz} → {root}")
        with tarfile.open(local_gz, "r:gz") as tar:
            tar.extractall(path=root.parent)  # extract into data/vendor/
        print(f"✅ Extracted into {root}")

    # 3. Scan & build manifest
    df = scan_monker(root)
    if df.empty:
        print(f"⚠️ No .txt files found under {root}")
        return

    # quick sanity print
    print(df.head(10).to_string(index=False))
    print(f"Total files indexed: {len(df)}")
    try:
        stacks = sorted(set(int(x) for x in df["stack_bb"].dropna().tolist()))
    except Exception:
        stacks = sorted(set(df["stack_bb"].dropna().tolist()))
    print("Distinct stacks:", stacks)
    print("Distinct hero_pos:", sorted(set(x for x in df["hero_pos"].dropna().tolist())))

    # drop rows missing opener_action
    df = df[df["opener_action"].notna()]

    write_manifest(df, out)

if __name__ == "__main__":
    main()
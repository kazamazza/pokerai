from __future__ import annotations
import argparse
import json
import hashlib
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.preflop.monker_helpers import canon_action, unique_seen_positions, classify_context, \
    parse_seq_from_stem, POS_SET, canon_pos
from infra.storage.s3_client import S3Client


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
        # skip macOS AppleDouble sidecars
        if path.name.startswith("._"):
            continue

        parts = list(path.parts)
        stack_bb = parse_stack_from_parts(parts)

        # hero folder = immediate parent
        hero_pos = canon_pos(path.parent.name)
        if hero_pos not in POS_SET:
            hero_pos = None

        stem = path.stem
        seq_raw = parse_seq_from_stem(stem)  # RAW vendor tokens (pos + action)
        # context (do NOT filter by SRP — keep everything)
        ctx_info = classify_context(seq_raw)

        # canonicalized sequence (for consistent downstream consumption)
        seq_canon = [
            {
                "pos": e["pos"],
                **({"action": canon_action(e.get("action"))} if "action" in e else {}),
            }
            for e in seq_raw
        ]

        # lightweight pair hints (optional): opener and unique positions
        opener_pos_raw, opener_action_raw = ctx_info["opener_pos_raw"], ctx_info["opener_action_raw"]
        pos_list = unique_seen_positions(seq_raw)

        # signatures & paths
        seq_json = json.dumps(seq_canon, sort_keys=True)
        file_sha1 = sha1_file(path)
        sig = sha1_str(f"{stack_bb}|{hero_pos}|{seq_json}")

        rows.append({
            "stack_bb": stack_bb,
            "hero_pos": hero_pos,
            # raw
            "sequence_raw": json.dumps(seq_raw, sort_keys=True),
            "opener_pos_raw": opener_pos_raw,
            "opener_action_raw": opener_action_raw,
            # canonical
            "sequence": seq_json,
            "opener_pos": opener_pos_raw,                      # positions are canonical already
            "opener_action": canon_action(opener_action_raw),  # RAISE/ALL_IN/etc.
            # ctx
            "ctx": ctx_info["ctx"],
            "raise_depth": ctx_info["raise_depth"],
            "limp_count": ctx_info["limp_count"],
            "multiway": ctx_info["multiway"],
            # optional hints
            "seen_positions": json.dumps(pos_list),
            # file identity
            "filename_stem": stem,
            "rel_path": str(path.relative_to(root)),
            "abs_path": str(path.resolve()),
            "file_sha1": file_sha1,
            "sig": sig,
        })

    df = pd.DataFrame(rows)

    # Dedup: keep one representative per (stack, hero, exact raw sequence)
    grouped = (
        df.groupby(["stack_bb", "hero_pos", "sequence_raw"], dropna=False)
          .agg(
              # canonical/ctx fields retained for downstream lookup
              opener_pos=("opener_pos", "first"),
              opener_action=("opener_action", "first"),
              opener_pos_raw=("opener_pos_raw", "first"),
              opener_action_raw=("opener_action_raw", "first"),
              ctx=("ctx", "first"),
              raise_depth=("raise_depth", "first"),
              limp_count=("limp_count", "first"),
              multiway=("multiway", "first"),
              seen_positions=("seen_positions", "first"),
              sequence=("sequence", "first"),
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
    ap.add_argument("--s3-bucket", type=str, default="pokeraistore")
    ap.add_argument("--s3-key", type=str, default="data/vendor/monker.tar.gz")
    ap.add_argument("--out", type=str, default="data/artifacts/monker_manifest.parquet")
    args = ap.parse_args()

    out = Path(args.out)

    # 1. Download tarball into tempdir
    s3c = S3Client()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        local_gz = tmpdir / "monker.tar.gz"
        s3c.download_file(args.s3_key, local_gz)

        # 2. Extract inside tempdir
        extract_root = tmpdir / "monker"
        print(f"▶️ Extracting {local_gz} → {extract_root}")
        with tarfile.open(local_gz, "r:gz") as tar:
            tar.extractall(path=extract_root)
        print(f"✅ Extracted into {extract_root}")

        # 3. Scan & build manifest
        df = scan_monker(extract_root)
        if df.empty:
            print(f"⚠️ No .txt files found under {extract_root}")
            return

        df = df[df["opener_action"].notna()]
        write_manifest(df, out)

    # tmpdir (tar + extracted files) auto-deleted here
    print("🧹 Temp dir cleaned up")

if __name__ == "__main__":
    main()
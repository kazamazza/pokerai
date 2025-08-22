from __future__ import annotations
import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

POS_NAMES = {
    "UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB", "EP", "MP", "BU"  # include common aliases
}

ACTION_NORMALIZE = {
    "AI": "ALL_IN",
    "Jam": "ALL_IN",
    "Allin": "ALL_IN",
    "Call": "CALL",
    "Raise": "RAISE",
    "Bet": "BET",
    "Check": "CHECK",
    "Cbet": "CBET",
    "Donk": "DONK",
    "Open": "OPEN",
    "Limp": "LIMP",
    "3Bet": "3BET",
    "4Bet": "4BET",
    "5Bet": "5BET",
    "Fold": "FOLD",
}

def normalize_action(tok: str) -> str:
    return ACTION_NORMALIZE.get(tok, tok)  # keep unknowns as-is

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_stack_from_parts(parts: List[str]) -> int | None:
    # find a path part like "12bb" (case-insensitive)
    for p in parts:
        q = p.lower()
        if q.endswith("bb"):
            try:
                return int(q[:-2])
            except ValueError:
                pass
    return None

def parse_filename_sequence(stem: str) -> List[Dict[str, str]]:
    """
    Parse names like: UTG_AI_HJ_Call_CO_Call_BTN_Call_SB_Call_BB_Call
    into: [{"pos":"UTG","action":"ALL_IN"}, {"pos":"HJ","action":"CALL"}, ...]
    We assume tokens alternate as POS, ACTION, POS, ACTION, ...
    If ACTION is missing after a POS, we record action=None.
    """
    toks = stem.split("_")
    seq: List[Dict[str, str]] = []
    i = 0
    while i < len(toks):
        pos = toks[i]
        if pos not in POS_NAMES:
            # Not a position? Just skip this token (robustness)
            i += 1
            continue
        action = None
        if i + 1 < len(toks) and toks[i + 1] not in POS_NAMES:
            action = normalize_action(toks[i + 1])
            i += 2
        else:
            i += 1
        entry = {"pos": pos}
        if action is not None:
            entry["action"] = action
        seq.append(entry)
    return seq


def scan_monker(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in root.rglob("*.txt"):
        # Expect structure like: datasets/vendor/monker/12bb/BB/<filename>.txt
        parts = list(path.parts)
        stack_bb = parse_stack_from_parts(parts)

        # infer hero position from immediate parent folder if it looks like a position
        hero_pos = path.parent.name
        if hero_pos not in POS_NAMES:
            hero_pos = None

        stem = path.stem  # filename without .txt
        seq = parse_filename_sequence(stem)

        opener_pos = seq[0]["pos"] if seq else None
        opener_action = seq[0].get("action") if seq else None

        rows.append({
            "stack_bb": stack_bb,  # int or None
            "hero_pos": hero_pos,  # str or None
            "opener_pos": opener_pos,  # str or None
            "opener_action": opener_action,  # str or None
            "sequence": json.dumps(seq),  # JSON string for readability
            "filename_stem": stem,  # original stem
            "rel_path": str(path.relative_to(root)),  # relative for portability
            "abs_path": str(path.resolve()),  # absolute path
            "sha1": sha1_file(path),  # content hash for dedup/integrity
        })

    df = pd.DataFrame(rows)

    # --- add aggregation with n_files ---
    # Keep one representative file for rel/abs path, but count all
    grouped = (
        df.groupby(["stack_bb", "hero_pos", "opener_pos", "opener_action"], dropna=False)
        .agg(
            sequence=("sequence", "first"),
            filename_stem=("filename_stem", "first"),
            rel_path=("rel_path", "first"),
            abs_path=("abs_path", "first"),
            sha1=("sha1", "first"),
            n_files=("rel_path", "count")
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
    ap.add_argument("--root", type=str, default="datasets/vendor/monker",
                    help="Root folder containing stack subfolders (e.g., 12bb, 15bb, ...)")
    ap.add_argument("--out", type=str, default="datasets/artifacts/monker_manifest.parquet")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    df = scan_monker(root)
    if df.empty:
        print(f"⚠️ No .txt files found under {root}")
        return

    # quick sanity print
    print(df.head(10).to_string(index=False))
    print(f"Total files indexed: {len(df)}")
    print("Distinct stacks:", sorted(set(df["stack_bb"].dropna().astype(int).tolist())))
    print("Distinct hero_pos:", sorted(set(x for x in df["hero_pos"].dropna().tolist())))

    write_manifest(df, out)

if __name__ == "__main__":
    main()
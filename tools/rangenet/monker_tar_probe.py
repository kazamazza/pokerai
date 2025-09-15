from __future__ import annotations
import argparse, collections, csv, os, re, tarfile
from pathlib import Path
from typing import Dict, List, Tuple

POS = ("UTG","HJ","CO","BTN","SB","BB")
POS_SET = set(POS)

# token patterns we care about
SIZE_X_RE   = re.compile(r"\b\d+(\.\d+)?x\b", re.IGNORECASE)  # 2.5x, 9x
PERCENT_RE  = re.compile(r"\b\d+%\b")                          # 60%
ACTION_RE   = re.compile(r"\b(open|raise|bet|3bet|4bet|5bet|cbet|donk|limp|call|fold)\b", re.IGNORECASE)
SEP_SPLIT   = re.compile(r"[ _\-\.\+]+")                       # common separators
CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")               # CamelCase boundaries

# HU pair patterns in directory or stem
PAIR_PATTERNS = [
    re.compile(r"\b(UTG|HJ|CO|BTN|SB|BB)[\s_\-]*vs[\s_\-]*(UTG|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE),
    re.compile(r"\b(UTG|HJ|CO|BTN|SB|BB)[\s_\-]*(?:v|x)[\s_\-]*(UTG|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE),
    re.compile(r"\b(UTG|HJ|CO|BTN|SB|BB)[\s_\-]+(UTG|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE),
]

def split_tokens(name: str) -> List[str]:
    base = name
    # split on separators then split CamelCase chunks again
    parts = []
    for tok in SEP_SPLIT.split(base):
        if not tok:
            continue
        parts.extend(CAMEL_SPLIT.split(tok))
    # normalize
    return [t for t in (p.strip() for p in parts) if t]

def find_positions(text: str) -> List[str]:
    hits = []
    for p in POS:
        if re.search(fr"\b{p}\b", text):
            hits.append(p)
    return hits

def find_hu_pair(text: str) -> Tuple[str|None, str|None]:
    for pat in PAIR_PATTERNS:
        m = pat.search(text)
        if m:
            a, b = m.group(1).upper(), m.group(2).upper()
            return a, b
    return None, None

def probe_tar(tar_path: Path, limit:int|None=None) -> Dict:
    exts = collections.Counter()
    lvl1 = collections.Counter()
    lvl2 = collections.Counter()
    stems_with_positions = 0
    stems_with_actions = 0
    stems_with_sizes = 0
    dir_pairs = collections.Counter()
    stem_pairs = collections.Counter()
    total_files = 0
    rows = []

    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isreg()]
        if limit:
            members = members[:limit]

        for m in members:
            total_files += 1
            path = m.name
            parts = Path(path).parts
            # extension
            ext = Path(path).suffix.lower()
            exts[ext] += 1
            # top dirs
            if len(parts) >= 2:
                lvl1[parts[0]] += 1
            if len(parts) >= 3:
                lvl2[os.path.join(parts[0], parts[1])] += 1

            name = Path(path).name
            stem = Path(path).stem

            # HU pairs from directories or stem
            dp_a, dp_b = find_hu_pair(path)
            if dp_a and dp_b:
                dir_pairs[(dp_a, dp_b)] += 1
            sp_a, sp_b = find_hu_pair(stem)
            if sp_a and sp_b:
                stem_pairs[(sp_a, sp_b)] += 1

            # tokens / flags
            toks = split_tokens(stem)
            text = " ".join(toks)
            pos_hits = find_positions(text)
            has_pos = len(pos_hits) > 0
            has_size = bool(SIZE_X_RE.search(text) or PERCENT_RE.search(text))
            has_action = bool(ACTION_RE.search(text))

            if has_pos:    stems_with_positions += 1
            if has_action: stems_with_actions += 1
            if has_size:   stems_with_sizes   += 1

            rows.append({
                "path": path,
                "ext": ext or "",
                "stem": stem,
                "has_pos": int(has_pos),
                "pos_hits": ",".join(sorted(set(pos_hits))),
                "has_action": int(has_action),
                "has_size": int(has_size),
                "hu_pair_dir": f"{dp_a}_vs_{dp_b}" if (dp_a and dp_b) else "",
                "hu_pair_stem": f"{sp_a}_vs_{sp_b}" if (sp_a and sp_b) else "",
            })

    return {
        "exts": exts,
        "lvl1": lvl1,
        "lvl2": lvl2,
        "stems_with_positions": stems_with_positions,
        "stems_with_actions": stems_with_actions,
        "stems_with_sizes": stems_with_sizes,
        "dir_pairs": dir_pairs,
        "stem_pairs": stem_pairs,
        "total_files": total_files,
        "rows": rows,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, type=str, help="Path to monker tar.gz")
    ap.add_argument("--out", required=True, type=str, help="CSV report path")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on scanned files")
    args = ap.parse_args()

    tar_path = Path(args.tar)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    res = probe_tar(tar_path, limit=args.limit)

    # Print summary
    print(f"\n=== MONKER TAR PROBE SUMMARY ===")
    print(f"Files scanned: {res['total_files']}")
    print("\nTop extensions:")
    for ext, c in res["exts"].most_common(12):
        print(f"  {ext or '(none)'}: {c}")

    print("\nTop level-1 directories:")
    for d, c in res["lvl1"].most_common(12):
        print(f"  {d}: {c}")

    print("\nTop level-2 directories:")
    for d, c in res["lvl2"].most_common(12):
        print(f"  {d}: {c}")

    print(f"\nStems w/ positions: {res['stems_with_positions']} / {res['total_files']}")
    print(f"Stems w/ action words: {res['stems_with_actions']} / {res['total_files']}")
    print(f"Stems w/ sizes (2.5x/60%): {res['stems_with_sizes']} / {res['total_files']}")

    if res["dir_pairs"]:
        print("\nHU pairs found in directories (top 10):")
        for (a,b), c in res["dir_pairs"].most_common(10):
            print(f"  {a} vs {b}: {c}")

    if res["stem_pairs"]:
        print("\nHU pairs found in stems (top 10):")
        for (a,b), c in res["stem_pairs"].most_common(10):
            print(f"  {a} vs {b}: {c}")

    # Write per-file CSV
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","ext","stem","has_pos","pos_hits","has_action","has_size","hu_pair_dir","hu_pair_stem"])
        w.writeheader()
        for row in res["rows"]:
            w.writerow(row)

    print(f"\n📄 Wrote probe CSV → {out}")
    print("Open it in a spreadsheet and filter by ext/hu_pair/has_pos/has_size to inspect quickly.\n")

if __name__ == "__main__":
    main()
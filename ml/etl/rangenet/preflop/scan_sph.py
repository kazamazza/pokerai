import argparse
from pathlib import Path
import json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/vendor/sph_norm")
    ap.add_argument("--out", default="data/artifacts/sph_manifest.parquet")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in root.rglob("*.json"):
        rec = json.loads(p.read_text())
        rows.append({
            "stack_bb": int(rec["stack_bb"]),
            "ip_pos": rec["ip_pos"],
            "oop_pos": rec["oop_pos"],
            "ctx": rec["ctx"],
            "source": "sph",
            "abs_path": str(p.resolve()),
            "rel_path": str(p.relative_to(root)),
        })
    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ sph_manifest → {out}  (rows: {len(df)})")

if __name__ == "__main__":
    main()
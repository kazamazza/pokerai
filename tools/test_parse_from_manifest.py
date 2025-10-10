import sys
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import gzip
import json
from pathlib import Path
import pandas as pd
from ml.etl.rangenet.postflop.solver_policy_parser import parse_solver_simple

# === CONFIG ===
LOCAL_SOLVER_DIR = Path("/Users/antoniocasamassa/S3/")
MANIFEST_PATH = Path("data/artifacts/rangenet_postflop_flop_manifest.parquet")
N_PER_CTX = 2  # how many examples to test per context

print("\n--- LOADING MANIFEST ---")
df = pd.read_parquet(MANIFEST_PATH)
print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

df = df[df["s3_key"].notna()]
contexts = df["ctx"].dropna().unique().tolist()
print(f"Found {len(contexts)} unique contexts:", contexts)

def decompress_if_needed(path: Path) -> Path:
    """Return a decompressed copy if .gz"""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        out_path = path.with_suffix("")  # strip .gz -> .json
        if not out_path.exists():
            with gzip.open(path, "rb") as fin, open(out_path, "wb") as fout:
                fout.write(fin.read())
        return out_path
    return path

def bet_sizes_from_manifest(v) -> list[float]:
    """
    Manifest column 'bet_sizes' is Avro-ish:
      None
      OR []
      OR [{'element': 0.33}, {'element': 0.66}, ...]
    Normalize to [0.33, 0.66] or [].
    """
    if v is None:
        return []
    if isinstance(v, list):
        if not v:
            return []
        # Avro list of records with 'element'
        if isinstance(v[0], dict) and "element" in v[0]:
            out = []
            for d in v:
                el = d.get("element", None)
                if el is not None:
                    try:
                        out.append(float(el))
                    except Exception:
                        pass
            return out
        # Already a list of floats (rare in your schema, but safe)
        try:
            return [float(x) for x in v]
        except Exception:
            return []
    # Unknown type → empty
    return []

results = []
for ctx in contexts:
    subset = df[df["ctx"] == ctx].head(N_PER_CTX)
    print(f"\n=== Context: {ctx} ({len(subset)} samples) ===")
    for _, row in subset.iterrows():
        s3_key = str(row["s3_key"])
        local_path = LOCAL_SOLVER_DIR / s3_key
        try:
            json_path = decompress_if_needed(local_path)
        except Exception as e:
            print(f"[skip] {s3_key}: {e}")
            continue

        # Pull numeric context from the manifest rather than guessing from JSON
        pot_bb = float(row.get("pot_bb") or 0.0)
        stack_bb = float(row.get("effective_stack_bb") or 100.0)
        menu_pcts = bet_sizes_from_manifest(row.get("bet_sizes"))  # -> list[float]
        if not menu_pcts:
            # fall back to size menu inferred from bet_sizing_id if you want,
            # but a safe default is fine for this sanity test:
            menu_pcts = [0.33, 0.66]

        try:
            # Use structured context in the parser
            probs_root, meta_root = parse_solver_simple(
                str(json_path),
                facing_bet=False,
                pot_bb=pot_bb,
                stack_bb=stack_bb,
                menu_pcts=tuple(menu_pcts),
            )
            probs_face, meta_face = parse_solver_simple(
                str(json_path),
                facing_bet=True,
                pot_bb=pot_bb,
                stack_bb=stack_bb,
                menu_pcts=tuple(menu_pcts),
            )

            def top3(probs: dict[str, float]) -> list[str]:
                items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                return [f"{k}:{round(v, 3)}" for k, v in items[:3] if v > 0]

            print(f"\n{s3_key}")
            print(f"  ip={row.get('ip_actor_flop')} | oop={row.get('oop_actor_flop')} | pot={pot_bb} | stack={stack_bb} | sizes={menu_pcts}")
            print(f"  ROOT : {top3(probs_root)}")
            print(f"  FACE : {top3(probs_face)}")

            results.append({
                "ctx": ctx,
                "s3_key": s3_key,
                "root_top3": top3(probs_root),
                "face_top3": top3(probs_face),
                "ip_pos": row.get("ip_actor_flop"),
                "oop_pos": row.get("oop_actor_flop"),
            })

        except Exception as e:
            print(f"[error] {s3_key}: {e}")
            continue

print("\n--- SUMMARY ---")
for ctx in contexts:
    n = sum(r["ctx"] == ctx for r in results)
    print(f"{ctx:15s}: {n} parsed OK")

print("\n✅ Test complete.")
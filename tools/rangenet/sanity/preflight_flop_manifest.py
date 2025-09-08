import argparse, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.build_rangenet_postflop_flop_manifest import build_manifest
from ml.utils.config import load_model_config

def main():
    ap = argparse.ArgumentParser(description="Preflight: ensure flop manifest has ranges for all rows.")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="model[/variant]/profile (same you use for build)")
    ap.add_argument("--out", type=str, default=None,
                    help="optional: write the manifest parquet here for inspection")
    args = ap.parse_args()

    cfg = load_model_config(model=args.config)
    print("building manifest",cfg)
    df = build_manifest(cfg)  # uses lookup.ranges_for_pair(strict=False)

    # Expect the builder to include a boolean flag 'ranges_missing'
    if "ranges_missing" not in df.columns:
        # derive it if needed: treat empty strings as missing
        miss = (df.get("range_ip","").astype(str) == "") | (df.get("range_oop","").astype(str) == "")
    else:
        miss = df["ranges_missing"].astype(bool)

    n_missing = int(miss.sum())
    n_total   = int(len(df))
    n_ok      = n_total - n_missing

    print(f"\nPreflight summary: total={n_total}  ok={n_ok}  missing={n_missing}")
    if n_missing > 0:
        print("\n❌ Missing ranges for these rows:")
        cols = ["positions","ctx","effective_stack_bb","pot_bb","board_cluster_id","range_ip_source_stack","range_oop_source_stack","range_ip_source_pair","range_oop_source_pair","range_ip_stack_delta","range_oop_stack_delta","range_ip_fallback_level","range_oop_fallback_level","s3_key"]
        cols = [c for c in cols if c in df.columns]
        print(df[miss][cols].head(50).to_string(index=False))
        print("\nTip: enable allow_pair_subs=True and sensible max_stack_delta; verify SPH/Monker manifests.")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.out, index=False)
            print(f"💾 wrote manifest with missing rows to {args.out}")
        sys.exit(1)

    print("✅ Preflight OK: every row has a range (after fallbacks).")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"💾 wrote manifest to {args.out}")

if __name__ == "__main__":
    main()
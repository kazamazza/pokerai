import sys
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from ml.utils.config import load_model_config


def main():
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(
        description="Preflop coverage summary for your manifest_build grid"
    )
    ap.add_argument("--config", default="rangenet/postflop",
                    help="model[/variant]/profile")
    ap.add_argument("--dump", type=str, default=None,
                    help="Optional path to write the detailed coverage CSV")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    mb = cfg["manifest_build"]; inputs = cfg["inputs"]; sv = cfg.get("solver", {})

    lookup = PreflopRangeLookup(
        monker_manifest_parquet="data/artifacts/monker_manifest.parquet",
        s3_client=S3Client(),                                # no lazy fetch in this report
        s3_prefix="data/vendor",
        cache_dir="data/vendor_cache"
    )

    rows = []
    for stack in mb["stacks_bb"]:
        nearest_stack = int(round(stack))  # for simple exact/nearest classification
        for ip, oop in [tuple(p) for p in mb["position_pairs"]]:
            rng_ip, rng_oop, meta = lookup.ranges_for_pair(
                stack_bb=stack, ip=ip, oop=oop, strict=False
            )

            if rng_ip and rng_oop:
                src_ip  = meta.get("range_ip_source_stack")
                src_oop = meta.get("range_oop_source_stack")
                if src_ip == nearest_stack and src_oop == nearest_stack:
                    cls = "exact"
                else:
                    cls = "nearest_stack"
            else:
                cls = "missing"

            rows.append({
                "pair": f"{ip}v{oop}",
                "stack": int(stack),
                "class": cls,
                "range_ip_source_stack": meta.get("range_ip_source_stack"),
                "range_oop_source_stack": meta.get("range_oop_source_stack"),
            })

    df = pd.DataFrame(rows)

    print("\nCoverage pivot (class by pair × stack):")
    # show the most frequent class per cell if multiple rows somehow exist
    if not df.empty:
        pivot = pd.pivot_table(
            df, index="pair", columns="stack", values="class",
            aggfunc=lambda x: x.value_counts().idxmax()
        )
        # make the stacks appear in numeric ascending order
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        print(pivot.fillna("").to_string())
    else:
        print("(no rows)")

    print("\nTotals:")
    if not df.empty:
        print(df["class"].value_counts().to_string())
    else:
        print("no data")

    if args.dump and not df.empty:
        out_path = args.dump
        df.to_csv(out_path, index=False)
        print(f"\n💾 wrote detailed coverage CSV → {out_path}")

if __name__=="__main__":
    main()
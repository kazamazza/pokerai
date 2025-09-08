# tools/rangenet/report_preflop_todo.py
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from ml.utils.config import load_model_config
# 6-max canonical order
ORDER = ["UTG","HJ","CO","BTN","SB","BB"]
IDX = {p:i for i,p in enumerate(ORDER)}

def dist(opener: str, defender: str) -> int:
    """# seats strictly between opener and defender (mod 6)"""
    return (IDX[defender] - IDX[opener] - 1) % len(ORDER)

def pairs_srp(monker_strict: bool) -> List[Tuple[str, str]]:
    # SRP opener → BB (drop SB→BB if aligning to Monker)
    opens = ["UTG","HJ","CO","BTN"] if monker_strict else ["UTG","HJ","CO","BTN","SB"]
    return [(op, "BB") for op in opens]

def pairs_limp_single(monker_strict: bool) -> List[Tuple[str, str]]:
    # Opener → (SB,BB), drop SB→BB if Monker-strict
    opens = ["UTG","HJ","CO","BTN"] if monker_strict else ["UTG","HJ","CO","BTN","SB"]
    pairs = []
    for op in opens:
        for df in ("SB","BB"):
            if monker_strict and op == "SB" and df == "BB":
                continue
            pairs.append((op, df))
    return pairs

def pairs_limp_multi() -> List[Tuple[str, str]]:
    # LIMP_SINGLE requires at least one seat in between ⇒ dist>=1
    # Valid defenders are SB or BB; exclude BTN→SB and SB→BB (dist=0).
    opens = ["UTG","HJ","CO","BTN"]  # SB opener to BB has dist=0, exclude
    pairs = []
    for op in opens:
        for df in ("SB","BB"):
            if dist(op, df) >= 1:
                pairs.append((op, df))
    # This yields: UTG→SB/BB, HJ→SB/BB, CO→SB/BB, BTN→BB (7 in total)
    return pairs

def expected_files(ctx: str, stack: int, ip: str, oop: str) -> List[str]:
    """
    Suggest exact raw files to paste (we still merge defender later).
    Context directories are SRP, LIMP_SINGLE, LIMP_SINGLE.
    """
    base = Path(f"data/vendor/sph/{ctx}/{stack}/{ip}_{oop}")
    files = []
    if ctx == "SRP":
        # IP opener open; OOP (BB) defend has call + two raises
        files = [
            str(base / "ip_open.txt"),
            str(base / "oop_call.txt"),
            str(base / "oop_raise_s1.txt"),
            str(base / "oop_raise_s2.txt"),
            # later you'll produce:
            str(base / "oop_defend.csv"),
        ]
    elif ctx == "LIMP_SINGLE":
        # Opener limps; defender (SB/BB) acts (call/raise sizes)
        files = [
            str(base / "ip_open.txt"),        # the limp range of the opener
            str(base / "oop_call.txt"),
            str(base / "oop_raise_s1.txt"),
            str(base / "oop_raise_s2.txt"),
            str(base / "oop_defend.csv"),
        ]
    elif ctx == "LIMP_SINGLE":
        # Opener limps; someone in the middle overlimps (we store as ip_call.txt);
        # defender (SB/BB) acts.
        files = [
            str(base / "ip_open.txt"),        # opener limp range
            str(base / "ip_call.txt"),        # caller overlimp range (pick a consistent caller for that pair)
            str(base / "oop_call.txt"),
            str(base / "oop_raise_s1.txt"),
            str(base / "oop_raise_s2.txt"),
            str(base / "oop_defend.csv"),
        ]
    else:
        files = []
    return files

def main():
    ap = argparse.ArgumentParser(description="Exact preflop chart TODOs (what you must solve/export).")
    ap.add_argument("--config", default="rangenet/preflop",
                    help="model[/variant]/profile that contains manifest_build grid")
    ap.add_argument("--monker-strict", action="store_true",
                    help="Drop SB→BB in SRP and Limped Single to align with typical Monker coverage")
    ap.add_argument("--dump", type=str, default=None,
                    help="Optional detailed CSV of TODOs")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    mb = cfg["manifest_build"]
    stacks = [int(s) for s in mb["stacks_bb"]]

    # Build the exact target grid we want:
    grid: List[Tuple[str,int,str,str]] = []  # (ctx, stack, ip, oop)
    for s in stacks:
        for (ctx, pairs_fn) in [
            ("SRP", lambda: pairs_srp(args.monker_strict)),
            ("LIMP_SINGLE", lambda: pairs_limp_single(args.monker_strict)),
            ("LIMP_SINGLE", pairs_limp_multi),
        ]:
            for (ip, oop) in pairs_fn():
                grid.append((ctx, s, ip, oop))

    # Use lookup to check if we already have ranges (Monker or SPH).
    lookup = PreflopRangeLookup(
        monker_manifest_parquet="data/artifacts/monker_manifest.parquet",
        s3_client=S3Client(),
        s3_vendor="data/vendor",
        cache_dir="data/vendor_cache",
        allow_pair_subs=False,   # set True if you want auto-substitution
    )

    rows = []
    for (ctx, s, ip, oop) in grid:
        rng_ip, rng_oop, meta = lookup.ranges_for_pair(
            stack_bb=s, ip=ip, oop=oop, ctx=ctx, strict=False
        )
        have = (rng_ip is not None) and (rng_oop is not None)
        files = expected_files(ctx, s, ip, oop)
        rows.append({
            "ctx": ctx,
            "stack": s,
            "pair": f"{ip}v{oop}",
            "status": "HAVE" if have else "MISSING",
            "source": meta.get("source"),
            "ip_src_stack": meta.get("range_ip_source_stack"),
            "oop_src_stack": meta.get("range_oop_source_stack"),
            "expected_files": " | ".join(files),
        })

    df = pd.DataFrame(rows)
    # Pretty print: group by status
    todo = df[df["status"] == "MISSING"].copy()
    have = df[df["status"] == "HAVE"].copy()

    if todo.empty:
        print("✅ All required charts present (SRP, LIMP_SINGLE, LIMP_SINGLE) for the configured stacks.")
    else:
        print("\n🚧 TODO — Solve & Export these charts:")
        print("(ctx, stack, pair) and the raw files you should paste")
        for _, r in todo.sort_values(["ctx","stack","pair"]).iterrows():
            print(f"  {r['ctx']:<12}  {r['stack']:>3}bb  {r['pair']:<8}  →  {r['expected_files']}")

    print("\nSummary:")
    print(df.groupby(["ctx","status"]).size().unstack(fill_value=0).to_string())

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.dump, index=False)
        print(f"\n💾 wrote detailed CSV → {args.dump}")

if __name__ == "__main__":
    main()
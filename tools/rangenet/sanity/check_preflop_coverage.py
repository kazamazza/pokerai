#!/usr/bin/env python3
import argparse, json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from ml.utils.config import load_model_config

# your imports


ORDER = ["UTG","HJ","CO","BTN","SB","BB"]
IDX = {p:i for i,p in enumerate(ORDER)}

def dist(opener: str, defender: str) -> int:
    return (IDX[defender] - IDX[opener] - 1) % len(ORDER)

def pairs_srp(monker_strict: bool) -> List[Tuple[str,str]]:
    opens = ["UTG","HJ","CO","BTN"] if monker_strict else ["UTG","HJ","CO","BTN","SB"]
    return [(op, "BB") for op in opens]

def pairs_limp_single(monker_strict: bool) -> list[tuple[str, str]]:
    # Single limper (opener) → defender in {SB, BB}, but SB→SB is NOT a thing.
    opens = ["UTG","HJ","CO","BTN"] if monker_strict else ["UTG","HJ","CO","BTN","SB"]
    pairs = []
    for op in opens:
        for df in ("SB","BB"):
            if op == "SB" and df == "SB":
                continue  # drop SB→SB
            if monker_strict and op == "SB" and df == "BB":
                continue  # optional: drop SB→BB if you want Monker alignment
            pairs.append((op, df))
    return pairs

def pairs_limp_multi() -> List[Tuple[str,str]]:
    opens = ["UTG","HJ","CO","BTN"]   # SB opener to BB is dist=0, exclude
    out = []
    for op in opens:
        for df in ("SB","BB"):
            if dist(op, df) >= 1:
                out.append((op, df))
    return out  # UTG/HJ/CO -> SB/BB (6) + BTN->BB (1) = 7

def _safe_len169(json_str: str) -> bool:
    try:
        arr = json.loads(json_str)
        return isinstance(arr, list) and len(arr) == 169 and all(isinstance(x,(int,float)) for x in arr)
    except Exception:
        return False

def _in_01(json_str: str) -> bool:
    arr = json.loads(json_str)
    return all(0.0 <= float(x) <= 1.0 for x in arr)

def _expected_files(ctx: str, stack: int, pair: str) -> str:
    """Construct the expected raw file list for a (ctx, stack, ip_v_oop) row."""
    ip, oop = pair.split("v")
    base = Path(f"data/vendor/sph/{ctx}/{int(stack)}/{ip}_{oop}")
    if ctx == "SRP":
        files = [
            base / "ip_open.txt",
            base / "oop_call.txt",
            base / "oop_raise_s1.txt",
            base / "oop_raise_s2.txt",
            base / "oop_defend.csv",
        ]
    elif ctx == "LIMP_SINGLE":
        files = [
            base / "ip_open.txt",
            base / "oop_call.txt",
            base / "oop_raise_s1.txt",
            base / "oop_raise_s2.txt",
            base / "oop_defend.csv",
        ]
    elif ctx == "LIMP_MULTI":
        files = [
            base / "ip_open.txt",
            base / "ip_call.txt",
            base / "oop_call.txt",
            base / "oop_raise_s1.txt",
            base / "oop_raise_s2.txt",
            base / "oop_defend.csv",
        ]
    else:
        files = []
    return " | ".join(str(p) for p in files)

def main():
    ap = argparse.ArgumentParser(description="Check preflop coverage & payload format (Monker+SPH).")
    ap.add_argument("--config", default="rangenet/preflop", help="model[/variant]/profile YAML key or path")
    ap.add_argument("--monker-strict", action="store_true", help="Align pairs with typical Monker coverage")
    ap.add_argument("--dump", type=Path, default=None, help="Optional CSV dump of per-row detail")
    ap.add_argument("--sph-manifest", type=Path, default=Path("data/artifacts/sph_manifest.parquet"),
                    help="SPH manifest parquet path (optional, but recommended)")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    mb = cfg.get("manifest_build") or cfg.get("build") or {}
    stacks = [int(s) for s in mb.get("stacks_bb", [25,60,100,150])]

    # Build target grid
    grid: List[Tuple[str,int,str,str]] = []
    for ctx, pf in [("SRP", lambda: pairs_srp(args.monker_strict)),
                    ("LIMP_SINGLE", lambda: pairs_limp_single(args.monker_strict)),
                    ("LIMP_MULTI", pairs_limp_multi)]:
        for (ip, oop) in pf():
            for s in stacks:
                grid.append((ctx, s, ip, oop))

    # Lookup that can use Monker + SPH (if manifest present)
    s3 = S3Client()
    lookup = PreflopRangeLookup(
        monker_manifest_parquet=cfg.get("inputs", {}).get("monker_manifest", "data/artifacts/monker_manifest.parquet"),
        s3_client=s3,
        s3_vendor=cfg.get("inputs", {}).get("vendor_s3_prefix", "data/vendor"),
        cache_dir=cfg.get("solver", {}).get("local_cache_dir", "data/vendor_cache"),
        allow_pair_subs=False,
        sph_manifest_parquet=str(args.sph_manifest) if args.sph_manifest and args.sph_manifest.exists() else None,
    )

    rows: List[Dict[str,Any]] = []
    missing = 0
    bad_format = 0

    for (ctx, s, ip, oop) in grid:
        ip_json, oop_json, meta = lookup.ranges_for_pair(stack_bb=s, ip=ip, oop=oop, ctx=ctx, strict=False)

        have = (ip_json is not None and oop_json is not None)
        fmt_ok = False
        rng_ok = False

        if have:
            fmt_ok = _safe_len169(ip_json) and _safe_len169(oop_json)
            rng_ok = fmt_ok and _in_01(ip_json) and _in_01(oop_json)

        if not have:
            missing += 1
        elif not rng_ok:
            bad_format += 1

        rows.append({
            "ctx": ctx,
            "stack": s,
            "pair": f"{ip}v{oop}",
            "status": "HAVE" if have else "MISSING",
            "format_ok": bool(fmt_ok),
            "range_in_0_1": bool(rng_ok),
            "source": meta.get("source"),
            "ip_src_stack": meta.get("range_ip_source_stack"),
            "oop_src_stack": meta.get("range_oop_source_stack"),
        })

    df = pd.DataFrame(rows)

    # ----------------- reporting -----------------
    print("\nCoverage (HAVE/MISSING) by ctx × stack:")
    if not df.empty:
        pivot = pd.pivot_table(
            df, index="ctx", columns="stack", values="status",
            aggfunc=lambda x: x.value_counts().idxmax()
        )
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        print(pivot.fillna("").to_string())
    else:
        print("(no rows)")

    print("\nTotals:")
    if not df.empty:
        print(df["status"].value_counts().to_string())
    else:
        print("no data")

    # Make sure the helper columns exist if you want to print them later
    if "expected_files" not in df.columns:
        df["expected_files"] = df.apply(
            lambda r: _expected_files(str(r["ctx"]), int(r["stack"]), str(r["pair"])), axis=1
        )

    # If you have a format check somewhere, keep it; else default to OK
    if "format_ok" not in df.columns:
        df["format_ok"] = True

    # Summaries
    missing_mask = (df["status"] == "MISSING")
    badfmt_mask = (df["format_ok"] == False)

    missing_n = int(missing_mask.sum())
    bad_format_n = int(badfmt_mask.sum())

    missing = df[missing_mask].copy()

    if not missing.empty:
        print("\n❌ Missing cells (ctx, stack, pair) and expected files:")
        for _, r in missing.sort_values(["ctx", "stack", "pair"]).iterrows():
            print(f"  {r['ctx']:<12}  {int(r['stack']):>3}bb  {r['pair']:<8}  →  {r['expected_files']}")

        print("\nHint: after merging + packing, scan should have produced canonical JSON under vendor_cache.")
        for _, r in missing.iterrows():
            ctx = str(r["ctx"]);
            s = int(r["stack"]);
            ip, oop = str(r["pair"]).split("v")
            expected_json = Path(f"data/vendor_cache/sph/{ctx}/{s}/{ip}_{oop}.json")
            if not expected_json.exists():
                print(f"  (no packed JSON) {expected_json}")

    # Final summary / exit code
    if missing_n == 0 and bad_format_n == 0:
        print("\n✅ 100% coverage and unified format OK (Monker+SPH both 169-float JSON).")
    else:
        if missing_n > 0:
            print(f"\n❌ Missing: {missing_n} row(s) need ranges.")
        if bad_format_n > 0:
            print(f"❌ Format issues: {bad_format_n} row(s) not in canonical 169-float form.")

if __name__ == "__main__":
    main()
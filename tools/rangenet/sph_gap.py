#!/usr/bin/env python3
"""
Report exactly which limped pairs (ctx, stack, IP, OOP) are required by your YAML settings
and which of those are present in SPH (both ip.csv & oop.csv) vs missing.

Usage:
  python tools/rangenet/report_limp_pairs.py \
    --settings config/your_settings.yaml \
    --sph data/artifacts/sph_manifest.parquet \
    --out-prefix data/artifacts/limp_pairs_report

Outputs:
  <out-prefix>_needed.csv
  <out-prefix>_have.csv         (rows where both heroes are present)
  <out-prefix>_missing.csv      (rows missing in SPH or missing one side)
"""

import argparse
from pathlib import Path
import itertools
import pandas as pd
import yaml

# --- helpers -------------------------------------------------------------

POS_ALIAS = {
    "ep":"UTG", "utg":"UTG",
    "mp":"HJ", "hj":"HJ",
    "co":"CO",
    "btn":"BTN",
    "sb":"SB",
    "bb":"BB",
}
def canon_pos(p: str) -> str:
    s = str(p).strip().upper()
    return POS_ALIAS.get(s.lower(), s)

def row_key(ctx, stack, ip, oop):
    return (str(ctx).upper(), int(stack), canon_pos(ip), canon_pos(oop))

def is_valid_limp_pair(ctx, ip, oop):
    ip, oop = ip.upper(), oop.upper()
    if ctx in ("LIMP_SINGLE","LIMP_MULTI"):
        if (ip, oop) == ("SB","BB"):   # SB limps, BB checks (HU)
            return True
        if (ip, oop) == ("BTN","BB"):  # BTN limps, SB folds, BB checks (rare but possible)
            return True
        return False
    return True

# --- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path,
                    help="YAML with manifest_build.scenarios[*]")
    ap.add_argument("--sph", required=True, type=Path,
                    help="sph_manifest.parquet produced by build_sph_manifest.py")
    ap.add_argument("--out-prefix", required=True, type=Path,
                    help="Prefix for CSV outputs (no extension)")
    args = ap.parse_args()

    # 1) Read YAML and enumerate required limped pairs
    cfg = yaml.safe_load(args.settings.read_text())
    scenarios = (cfg.get("manifest_build") or {}).get("scenarios", [])

    LIMPS = {"LIMPED_SINGLE", "LIMP_SINGLE", "LIMPED_MULTI", "LIMP_MULTI"}

    need_rows = []
    for sc in scenarios:
        ctx = str(sc.get("ctx","")).upper()
        if ctx not in LIMPS:
            continue
        # normalize ctx names to our SPH/lookup convention
        ctx_norm = "LIMP_SINGLE" if "SINGLE" in ctx else "LIMP_MULTI"

        stacks = sc.get("stacks_bb") or sc.get("stacks") or []
        pairs  = sc.get("position_pairs") or []
        # pairs are like [["BB","SB"], ["BTN","BB"], ...]
        for stack, (ip, oop) in itertools.product(stacks, pairs):
            need_rows.append({
                "ctx": ctx_norm,
                "stack": int(stack),
                "ip": canon_pos(ip),
                "oop": canon_pos(oop),
            })

    need = (pd.DataFrame(need_rows)
            .drop_duplicates()
            .sort_values(["ctx","stack","ip","oop"])
            .reset_index(drop=True))

    need = need[need.apply(lambda r: is_valid_limp_pair(r["ctx"], r["ip"], r["oop"]), axis=1)]

    if need.empty:
        raise SystemExit("No limped scenarios found in YAML (LIMPED_SINGLE/LIMPED_MULTI).")

    # 2) Load SPH manifest and compute which (ctx,stack,ip,oop) have both sides
    sph = pd.read_parquet(args.sph).copy()
    # normalize columns just in case
    sph["ctx"]      = sph["ctx"].map(lambda s: str(s).upper())
    sph["stack_bb"] = pd.to_numeric(sph["stack_bb"], errors="coerce").astype("Int64")
    sph["ip_pos"]   = sph["ip_pos"].map(canon_pos)
    sph["oop_pos"]  = sph["oop_pos"].map(canon_pos)
    sph["hero_pos"] = sph["hero_pos"].map(lambda s: canon_pos(s) or str(s).upper())

    # Accept both "LIMPED_*" and "LIMP_*" in SPH (packer used LIMP_SINGLE/LIMP_MULTI)
    def canon_ctx_sph(c: str) -> str:
        c = str(c).upper()
        if c == "LIMPED_SINGLE": return "LIMP_SINGLE"
        if c == "LIMPED_MULTI":  return "LIMP_MULTI"
        return c

    sph["ctx_norm"] = sph["ctx"].map(canon_ctx_sph)
    have = (sph.groupby(["ctx_norm","stack_bb","ip_pos","oop_pos"])["hero_pos"]
                .apply(lambda s: set(s)=={"IP","OOP"})
                .reset_index(name="has_both")
                .rename(columns={"ctx_norm":"ctx","stack_bb":"stack","ip_pos":"ip","oop_pos":"oop"}))

    # 3) Join NEED with HAVE and split into have/missing
    need_cov = need.merge(have, how="left", on=["ctx","stack","ip","oop"])
    need_cov["has_both"] = need_cov["has_both"].fillna(False)

    have_both   = need_cov[need_cov["has_both"]==True].copy()
    missing_any = need_cov[need_cov["has_both"]==False].copy()

    # 4) Pretty console output
    def banner(t): print("\n" + t + "\n" + "-"*len(t))
    banner("Required limped pairs (from YAML)")
    print(need.to_string(index=False))

    banner("Coverage summary by ctx & stack (SPH)")
    summ = (need_cov
            .groupby(["ctx","stack"])
            .agg(required=("ctx","size"),
                 have=("has_both","sum"))
            .reset_index())
    summ["missing"] = summ["required"] - summ["have"]
    print(summ.sort_values(["ctx","stack"]).to_string(index=False))

    if not missing_any.empty:
        banner("Missing (need but SPH lacks ip and/or oop)")
        print(missing_any.sort_values(["ctx","stack","ip","oop"]).to_string(index=False))
    else:
        print("\nAll required limped pairs are covered in SPH. ✅")

    # 5) Write CSVs
    out_pref = args.out_prefix
    need.to_csv(f"{out_pref}_needed.csv", index=False)
    have_both.to_csv(f"{out_pref}_have.csv", index=False)
    missing_any.to_csv(f"{out_pref}_missing.csv", index=False)
    print(f"\nWrote:\n  {out_pref}_needed.csv\n  {out_pref}_have.csv\n  {out_pref}_missing.csv")


if __name__ == "__main__":
    main()
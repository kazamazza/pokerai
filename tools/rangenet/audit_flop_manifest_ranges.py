import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Set

import pandas as pd
import yaml


def canon_pos(p: str) -> str:
    return str(p).strip().upper() if p else ""

def parse_positions_col(s: str) -> Tuple[str, str]:
    # expects like "BTNvBB"
    try:
        a, b = str(s).split("v", 1)
        return canon_pos(a), canon_pos(b)
    except Exception:
        return "", ""

def add_ip_oop_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"ip_actor_flop","oop_actor_flop"}.issubset(df.columns):
        df["ip_pos"]  = df["ip_actor_flop"].map(canon_pos)
        df["oop_pos"] = df["oop_actor_flop"].map(canon_pos)
    elif "positions" in df.columns:
        ip_oop = df["positions"].map(parse_positions_col)
        df["ip_pos"]  = ip_oop.map(lambda t: t[0])
        df["oop_pos"] = ip_oop.map(lambda t: t[1])
    else:
        df["ip_pos"] = ""
        df["oop_pos"] = ""
    return df

def load_allowed_pairs_from_yaml(settings_path: Path) -> Dict[str, Set[Tuple[str,str]]]:
    """
    Reads YAML and returns allowed pairs per limped ctx, e.g.:
    { "LIMPED_SINGLE": {("BTN","BB")}, "LIMPED_MULTI": {("BTN","BB")} }
    Falls back to BTN-BB for both if not found.
    """
    default = {"LIMPED_SINGLE": {("BTN","BB")}, "LIMPED_MULTI": {("BTN","BB")}}
    if not settings_path or not settings_path.exists():
        return default

    with open(settings_path, "r") as f:
        cfg = yaml.safe_load(f)

    out: Dict[str, Set[Tuple[str,str]]] = {"LIMPED_SINGLE": set(), "LIMPED_MULTI": set()}
    # Walk scenarios if present
    scenarios = cfg.get("scenarios") or cfg.get("jobs") or []
    for sc in scenarios:
        ctx = str(sc.get("ctx") or sc.get("name") or "").upper()
        if ctx not in out:
            continue
        pairs = sc.get("position_pairs") or sc.get("position_pairs_hu") or []
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                out[ctx].add((canon_pos(pair[0]), canon_pos(pair[1])))

    # Apply defaults if any set is empty
    for k in out:
        if not out[k]:
            out[k] = default[k]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--settings", type=Path, default=None, help="YAML settings to derive legal limped pairs")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    if "ctx" not in df.columns:
        print("manifest missing 'ctx' column")
        sys.exit(2)

    df["ctx"] = df["ctx"].map(lambda s: str(s).upper())
    df = add_ip_oop_cols(df)

    # Determine allowed limped pairs
    allowed = load_allowed_pairs_from_yaml(args.settings) if args.settings else {
        "LIMPED_SINGLE": {("BTN","BB")},
        "LIMPED_MULTI":  {("BTN","BB")},
    }

    # Filter out illegal limped rows (diagnose)
    limped_mask = df["ctx"].isin(["LIMPED_SINGLE","LIMPED_MULTI"])
    before = limped_mask.sum()
    def is_legal(row) -> bool:
        if row["ctx"] not in allowed:
            return True  # non-limped contexts unaffected
        return (row["ip_pos"], row["oop_pos"]) in allowed[row["ctx"]]

    df_legal = df[~limped_mask | df.apply(is_legal, axis=1)].copy()
    after = (df_legal["ctx"].isin(["LIMPED_SINGLE","LIMPED_MULTI"])).sum()
    dropped = before - after
    if dropped:
        print(f"Ignored illegal limped rows: {dropped} (kept {after} / total limped {before})")

    # Summary by ctx/stack/source
    print("Range sources by ctx and stack:")
    summ = (
        df_legal
        .groupby(["ctx","effective_stack_bb","range_source"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(summ)

    # Monker-backed examples (unchanged)
    print("\nExamples that should be Monker-backed:")
    for ctx in ["VS_OPEN","BLIND_VS_STEAL","VS_3BET","VS_4BET"]:
        for st in [60.0, 100.0, 150.0]:
            sub = df_legal[(df_legal.ctx==ctx) & (df_legal.effective_stack_bb==st)]
            srcs = sub["range_source"].value_counts()
            print(f"{ctx} @{int(st)}bb → {dict(srcs)}")

    # Show a few real rows (monker or sph)
    print("\nSample non-fallback rows:")
    mask = df_legal["range_source"].str.startswith(("monker","sph"))
    cols = [c for c in ["ctx","topology","effective_stack_bb","pot_bb","positions",
                        "bet_sizing_id","range_source","ip_pos","oop_pos"] if c in df_legal.columns]
    print(df_legal[mask].head(20)[cols])

if __name__ == "__main__":
    main()
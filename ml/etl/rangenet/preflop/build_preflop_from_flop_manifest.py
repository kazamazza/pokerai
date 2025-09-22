#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# === Config ===
SCENARIO_KEYS = ["stack_bb", "hero_pos", "opener_pos", "opener_action", "ctx"]

# Your canonical 169 ordering
ALL_HANDS = [...]  # e.g. ["AA","KK","QQ",...,"32o"]
HAND_TO_ID = {h: i for i, h in enumerate(ALL_HANDS)}

_POS = {"UTG","HJ","CO","BTN","SB","BB","IP","OOP"}

def canon_pos(p: str) -> str:
    s = str(p).upper()
    return s if s in _POS else s

def ctx_from_manifest(row: pd.Series) -> str:
    # Prefer explicit ctx if present, else infer from menu_id or topology
    if "ctx" in row and pd.notna(row["ctx"]):
        return str(row["ctx"]).upper()
    m = str(row.get("bet_sizing_id","")).lower()
    if m.startswith("limped_single"): return "LIMPED_SINGLE"
    if m.startswith("limped_multi"):  return "LIMPED_MULTI"
    if m.startswith("3bet"):          return "VS_3BET"
    if m.startswith("4bet"):          return "VS_4BET"
    return "SRP"

def opener_action_for_ctx(ctx: str) -> str:
    c = str(ctx).upper()
    if c in ("SRP","BLIND_VS_STEAL"): return "RAISE"
    if c == "VS_3BET": return "3BET"
    if c == "VS_4BET": return "4BET"
    if c in ("LIMPED_SINGLE","LIMPED_MULTI"): return "LIMP"
    return "RAISE"

def positions_from_manifest(row: pd.Series) -> Tuple[str,str]:
    # Expect "positions" like "BTNvBB" with opener v defender semantics (as in your manifests)
    pos = str(row.get("positions","")).upper()
    if "V" in pos:
        a,b = pos.split("V",1)
        return canon_pos(a), canon_pos(b)
    # very conservative fallback
    return "BTN","BB"

def to_vec169(rng) -> np.ndarray:
    # accepts dict, list(169), 13x13, or JSON string of those
    if isinstance(rng, str):
        try: rng = json.loads(rng)
        except Exception: rng = {}
    if isinstance(rng, (list, tuple, np.ndarray)):
        arr = np.asarray(rng, dtype="float32")
        if arr.ndim == 2 and arr.shape == (13,13):
            v = arr.reshape(169).astype("float32")
        elif arr.ndim == 1 and arr.size == 169:
            v = arr.astype("float32")
        else:
            v = np.zeros(169, dtype="float32")
    elif isinstance(rng, dict):
        v = np.zeros(169, dtype="float32")
        for h,w in rng.items():
            idx = HAND_TO_ID.get(str(h))
            if idx is not None:
                try: v[idx] = float(w)
                except: pass
    else:
        v = np.zeros(169, dtype="float32")
    s = float(v.sum())
    return (v/s) if s > 0 else np.ones(169, dtype="float32")/169.0

def build_preflop_from_flop_manifest(
    flop_manifest: Path,
    out_parquet: Path,
    root_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_parquet(flop_manifest)

    # keep “preflop root” rows only
    if root_only:
        if "street" in df.columns:
            df = df[df["street"].astype(int) == 1]
        if "node_key" in df.columns:
            df = df[df["node_key"].astype(str) == "root"]
        df = df.reset_index(drop=True)

    # derive scenario keys per-row
    stacks = np.rint(df.get("effective_stack_bb").astype(float)).astype(int)
    opener_pos, hero_pos = zip(*df.apply(positions_from_manifest, axis=1))
    ctx_vals = df.apply(ctx_from_manifest, axis=1)
    open_act = [opener_action_for_ctx(c) for c in ctx_vals]

    df_keys = pd.DataFrame({
        "stack_bb": stacks,
        "opener_pos": list(opener_pos),
        "hero_pos":   list(hero_pos),
        "ctx":        ctx_vals.astype(str),
        "opener_action": open_act,
    })

    # filter rows that have both ranges
    has_ip  = df["range_ip"].notna()
    has_oop = df["range_oop"].notna()
    df = df[has_ip & has_oop].reset_index(drop=True)
    df_keys = df_keys.loc[df.index].reset_index(drop=True)

    # attach keys back
    df = pd.concat([df_keys, df[["range_ip","range_oop"]]], axis=1)

    # group by scenario & average villain vectors
    rows: List[Dict[str, float]] = []
    for keys, g in df.groupby(SCENARIO_KEYS):
        # determine villain side per row, convert to vec, then mean
        vecs = []
        for _, r in g.iterrows():
            op_pos = r["opener_pos"]; hero = r["hero_pos"]
            ip, oop = positions_from_manifest(pd.Series({"positions": f"{op_pos}v{hero}"}))
            # if hero is IP, villain is OOP; else villain is IP
            target = r["range_oop"] if hero == ip else r["range_ip"]
            vecs.append(to_vec169(target))
        y = np.mean(np.stack(vecs, axis=0), axis=0).astype("float32")

        row = {k: v for k, v in zip(SCENARIO_KEYS, keys)}
        row["weight"] = float(len(g))
        for i, val in enumerate(y.tolist()):
            row[f"y_{i}"] = float(val)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(f"✅ wrote {out_parquet} with {len(out):,} scenarios (from {len(df):,} root rows)")
    return out

def main():
    ap = argparse.ArgumentParser("Build preflop dataset directly from postflop flop manifest")
    ap.add_argument("--flop-manifest", default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--out", default="data/datasets/rangenet_preflop_from_flop.parquet")
    ap.add_argument("--all-nodes", action="store_true", help="include non-root rows (averaged by scenario)")
    args = ap.parse_args()

    build_preflop_from_flop_manifest(
        Path(args.flop_manifest),
        Path(args.out),
        root_only=not args.all_nodes,
    )

if __name__ == "__main__":
    main()
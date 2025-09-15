#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# --- minimal helpers (match your project’s conventions) ---
Pair = Tuple[str, str]

CTX_ALIAS = {
    "OPEN":"SRP","VS_OPEN":"SRP","VS_OPEN_RFI":"SRP",
    "3BET":"VS_3BET","VS_3BET":"VS_3BET",
    "4BET":"VS_4BET","VS_4BET":"VS_4BET",
    "BVS":"BLIND_VS_STEAL","BLIND_VS_STEAL":"BLIND_VS_STEAL",
    "LIMPED_SINGLE":"LIMP_SINGLE","LIMP_SINGLE":"LIMP_SINGLE",
    "LIMPED_MULTI":"LIMP_MULTI","LIMP_MULTI":"LIMP_MULTI",
}

def canon_ctx(c: str) -> str:
    c2 = str(c).upper()
    return CTX_ALIAS.get(c2, c2)

def canon_pos(p: str|None) -> Optional[str]:
    if not p: return None
    q = str(p).strip().upper()
    return q if q in {"UTG","HJ","CO","BTN","SB","BB","IP","OOP"} else None

def canon_pair(ip: str, oop: str) -> Optional[Pair]:
    a, b = canon_pos(ip), canon_pos(oop)
    if not a or not b or a == b: return None
    return (a, b)

# Keep this aligned with your sanitizer rules
VALID: Dict[str, Set[Pair]] = {
    "SRP": {
        ("UTG","BB"),("HJ","BB"),("CO","BB"),("BTN","BB"),
        ("BTN","SB"),("BB","SB"),("UTG","SB"),("HJ","SB"),("CO","SB"),
    },
    "BLIND_VS_STEAL": {("BTN","BB"),("BTN","SB"),("CO","BB")},
    "VS_3BET": {
        ("BTN","BB"),("CO","BB"),("HJ","BB"),("UTG","BB"),
        ("BTN","SB"),("CO","SB"),("HJ","SB"),
        ("BB","BTN"),("SB","BTN"),("CO","BTN"),
    },
    "VS_4BET": {("BTN","BB"),("BB","BTN"),("BTN","SB"),("SB","BTN")},
    "LIMP_SINGLE": {("BB","SB")},
    "LIMP_MULTI": {("BB","SB"),("BTN","SB")},
}

def sanitize_pairs(pairs: List[Pair], ctx: str) -> List[Pair]:
    ctx2 = canon_ctx(ctx)
    legal = VALID.get(ctx2, set())
    out, seen = [], set()
    for ip, oop in pairs:
        cp = canon_pair(ip, oop)
        if not cp: continue
        if legal and cp not in legal: continue
        if cp not in seen:
            seen.add(cp); out.append(cp)
    return out

# --- auditor ---
def load_monker_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    need = {"stack_bb","hero_pos","ctx","ip_pos","oop_pos","rel_path"}
    miss = [c for c in need if c not in df.columns]
    if miss: raise RuntimeError(f"Monker manifest missing: {miss}")

    df["ctx"] = df["ctx"].map(lambda s: canon_ctx(s) if s is not None else None)
    for c in ("hero_pos","ip_pos","oop_pos"):
        df[c] = df[c].map(canon_pos)
    df = df.dropna(subset=["ctx","hero_pos","ip_pos","oop_pos","stack_bb"])
    df["stack_bb"] = df["stack_bb"].astype("Int64")
    return df

def load_sph_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    need = {"stack_bb","ctx","ip_pos","oop_pos","hero_pos","rel_path","abs_path"}
    miss = [c for c in need if c not in df.columns]
    if miss: raise RuntimeError(f"SPH manifest missing: {miss}")

    df["ctx"] = df["ctx"].map(lambda s: canon_ctx(s))
    for c in ("hero_pos","ip_pos","oop_pos"):
        df[c] = df[c].map(canon_pos)
    df = df.dropna(subset=["ctx","hero_pos","ip_pos","oop_pos","stack_bb"])
    df["stack_bb"] = df["stack_bb"].astype("Int64")
    return df

def build_available_set(monker_df: pd.DataFrame, sph_df: Optional[pd.DataFrame]) -> Dict[Tuple[str,int,Pair], str]:
    """
    Return mapping (ctx, stack, (ip,oop)) -> source tag "monker"|"sph"|"both"
    We require BOTH IP and OOP files to consider pair available per source.
    """
    avail: Dict[Tuple[str,int,Pair], Set[str]] = {}

    # Monker: require both hero=IP and hero=OOP present for same (ctx,stack,ip,oop)
    mkey_cols = ["ctx","stack_bb","ip_pos","oop_pos"]
    grp = monker_df.groupby(mkey_cols)
    for key, g in grp:
        # check that both hero sides exist
        if set(g["hero_pos"].unique()) >= {"IP","OOP"}:
            ctx, stack, ip, oop = key
            k = (ctx, int(stack), (ip, oop))
            avail.setdefault(k, set()).add("monker")

    if sph_df is not None and not sph_df.empty:
        grp2 = sph_df.groupby(mkey_cols)
        for key, g in grp2:
            if set(g["hero_pos"].unique()) >= {"IP","OOP"}:
                ctx, stack, ip, oop = key
                k = (ctx, int(stack), (ip, oop))
                avail.setdefault(k, set()).add("sph")

    # collapse to tag
    out: Dict[Tuple[str,int,Pair], str] = {}
    for k, srcs in avail.items():
        out[k] = "both" if len(srcs) > 1 else next(iter(srcs))
    return out

def expand_required_from_config(cfg: dict) -> Set[Tuple[str,int,Pair]]:
    mb = cfg.get("manifest_build", {}) or {}
    scenarios = mb.get("scenarios") or []
    if not scenarios:
        # fallback single scenario if none configured
        scenarios = [{
            "name": str(mb.get("ctx","SRP")),
            "ctx":  str(mb.get("ctx","SRP")),
            "stacks_bb": mb.get("stacks_bb", [100]),
            "position_pairs": mb.get("position_pairs", [("BTN","BB")]),
        }]

    required: Set[Tuple[str,int,Pair]] = set()
    for sc in scenarios:
        ctx = canon_ctx(sc.get("ctx","SRP"))
        stacks = [int(float(x)) for x in sc.get("stacks_bb", [100])]
        raw_pairs = [(str(a), str(b)) for (a,b) in sc.get("position_pairs", [("BTN","BB")])]
        pairs = sanitize_pairs(raw_pairs, ctx)
        for s in stacks:
            for (ip, oop) in pairs:
                required.add((ctx, s, (ip, oop)))
    return required

def audit(monker_pq: Path, sph_pq: Optional[Path], cfg: dict) -> dict:
    monker_df = load_monker_manifest(monker_pq)
    sph_df = load_sph_manifest(sph_pq) if (sph_pq and sph_pq.exists()) else None

    available = build_available_set(monker_df, sph_df)
    required = expand_required_from_config(cfg)

    have = set(available.keys())
    missing = sorted(required - have)
    extra   = sorted(have - required)

    # scenario-level rollups
    by_ctx: Dict[str, Dict[str,int]] = {}
    for (ctx, s, pair) in required:
        d = by_ctx.setdefault(ctx, {"required":0,"available":0})
        d["required"] += 1
        if (ctx, s, pair) in have:
            d["available"] += 1

    # per-pair source stats
    src_counts: Dict[str,int] = {"monker":0,"sph":0,"both":0}
    for k, tag in available.items():
        if k in required:
            src_counts[tag] += 1

    return {
        "summary": {
            "required_total": len(required),
            "available_total": len(have & required),
            "missing_total": len(missing),
            "coverage_pct": round(100.0 * (len(have & required) / max(1, len(required))), 2),
            "source_breakdown_for_required": src_counts,
        },
        "by_ctx": by_ctx,
        "missing": [
            {"ctx":ctx,"stack":stack,"ip":ip,"oop":oop} for (ctx, stack, (ip, oop)) in missing
        ],
        "extra_available_not_required": [
            {"ctx":ctx,"stack":stack,"ip":ip,"oop":oop,"source":available[(ctx,stack,(ip,oop))]}
            for (ctx,stack,(ip,oop)) in extra
        ],
    }

def main():
    ap = argparse.ArgumentParser(description="Audit preflop coverage (Monker + SPH) vs desired scenarios.")
    ap.add_argument("--monker", type=Path, default=Path("data/artifacts/monker_manifest.parquet"))
    ap.add_argument("--sph", type=Path, default=Path("data/artifacts/sph_manifest.parquet"))
    ap.add_argument("--config", type=Path, default=Path("ml/config/rangenet/postflop/prod.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/artifacts/preflop_coverage.json"))
    args = ap.parse_args()

    # load your YAML/JSON config; replace with your loader if needed
    import yaml
    cfg = yaml.safe_load(args.config.read_text())

    report = audit(args.monker, args.sph, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote coverage → {args.out}")

if __name__ == "__main__":
    main()
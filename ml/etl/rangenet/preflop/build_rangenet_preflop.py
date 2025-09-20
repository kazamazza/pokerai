from __future__ import annotations
import sys
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.config.types_hands import ALL_HANDS, HAND_TO_ID
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from ml.utils.config import load_model_config
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- project imports you already have ---

# ====================== helpers ======================

SCENARIO_KEYS = ["stack_bb", "hero_pos", "opener_pos", "opener_action", "ctx"]

_POS = {"UTG","HJ","CO","BTN","SB","BB","IP","OOP"}

def _load_ctx_backoff(cfg: Dict[str, Any]) -> Dict[str, str]:
    raw = ((cfg.get("manifest_build", {}) or {}).get("ctx_backoff", {}) or {})
    # normalize to UPPER
    return {str(k).upper(): str(v).upper() for k, v in raw.items() if v}

def _canon_pos(p: str) -> str:
    s = str(p).upper()
    return s if s in _POS else s

def _range_map_to_vector(rng_map: Dict[str, float] | str | None) -> np.ndarray:
    """
    Convert a hand→weight dict (or JSON string) into a dense 169-dim float32 vector.
    Ensures L1-normalized; uniform fallback if empty.
    """
    v = np.zeros(len(ALL_HANDS), dtype="float32")
    if isinstance(rng_map, str):
        try: rng_map = json.loads(rng_map)
        except Exception: rng_map = {}
    if isinstance(rng_map, dict):
        for h, w in rng_map.items():
            idx = HAND_TO_ID.get(str(h))
            if idx is not None:
                try: v[idx] = float(w)
                except Exception: pass
    s = float(v.sum())
    if s > 0: v /= s
    else:     v[:] = 1.0 / len(ALL_HANDS)
    return v

def _ctx_alias(c: str) -> str:
    return PreflopRangeLookup.CTX_ALIAS.get(str(c).upper(), str(c).upper())

def _opener_action_for_ctx(ctx: str) -> str:
    c = _ctx_alias(ctx)
    if c in ("SRP", "BLIND_VS_STEAL"): return "RAISE"
    if c == "VS_3BET":                 return "3BET"
    if c == "VS_4BET":                 return "4BET"
    if c in ("LIMPED_SINGLE","LIMPED_MULTI"): return "LIMP"
    return "RAISE"

def _candidate_ip_oop(opener_pos: str, hero_pos: str) -> List[Tuple[str,str]]:
    """
    Preflop semantic guess for who is IP/OOP postflop, given opener & hero.
    We try a couple of plausible mappings and let the lookup fill gaps/substitute.
    """
    a = _canon_pos(opener_pos); b = _canon_pos(hero_pos)
    out: List[Tuple[str,str]] = []
    # common: opener is IP vs blinds
    if b == "BB": out.append((a,b))
    # SB vs BB steal
    if a == "SB" and b == "BB": out.append((a,b))
    # generic opener-IP first, then swapped
    out.extend([(a,b), (b,a)])
    # de-dup preserve order
    seen = set(); uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def _lookup_vec169(
    lookup: PreflopRangeLookup,
    *,
    stack_bb: int,
    hero_pos: str,
    opener_pos: str,
    ctx: str,
) -> Optional[np.ndarray]:
    """
    Ask lookup for (rng_ip, rng_oop). Return *villain* range as 169-vector.
    Villain = opponent of hero.
    """
    hero  = _canon_pos(hero_pos)
    openr = _canon_pos(opener_pos)
    cctx  = _ctx_alias(ctx)

    for ip, oop in _candidate_ip_oop(openr, hero):
        try:
            rng_ip, rng_oop, _meta = lookup.ranges_for_pair(
                stack_bb=stack_bb,
                ip=ip, oop=oop,
                ctx=cctx,
                strict=False
            )
        except Exception:
            continue

        if hero == ip:
            target = rng_oop
        elif hero == oop:
            target = rng_ip
        else:
            # fall back: take the side opposite the opener
            target = rng_oop if openr == ip else rng_ip

        return _range_map_to_vector(target)
    return None

# ====================== core builder ======================

def _expand_scenarios_from_yaml(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Read manifest_build.scenarios from YAML and expand into rows:
    (stack_bb, opener_pos, hero_pos, opener_action, ctx).
    """
    sc_list = (cfg.get("manifest_build", {}) or {}).get("scenarios", []) or []
    rows: List[Dict[str, Any]] = []

    for sc in sc_list:
        ctx = sc.get("ctx") or "SRP"
        stacks = sc.get("stacks_bb") or []
        pos_pairs = sc.get("position_pairs") or []
        opener_action = _opener_action_for_ctx(ctx)

        for st in stacks:
            st_int = int(round(float(st)))
            for pair in pos_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                opener_pos = _canon_pos(pair[0])
                hero_pos   = _canon_pos(pair[1])   # defender by convention
                rows.append({
                    "stack_bb": st_int,
                    "opener_pos": opener_pos,
                    "hero_pos": hero_pos,
                    "opener_action": opener_action,
                    "ctx": _ctx_alias(ctx),
                })

    if not rows:
        return pd.DataFrame(columns=SCENARIO_KEYS)
    df = pd.DataFrame(rows, columns=SCENARIO_KEYS)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def build_rangenet_preflop_from_cfg(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Build preflop dataset by expanding scenario grid and querying PreflopRangeLookup.
    """
    # Inputs
    monker_manifest = Path(cfg["inputs"]["manifest_monker"])
    sph_manifest    = Path(cfg["inputs"].get("manifest_sph","")) if cfg["inputs"].get("manifest_sph") else None
    out_parquet     = Path(cfg["outputs"].get("parquet","data/datasets/rangenet_preflop.parquet"))

    if not monker_manifest.exists():
        raise FileNotFoundError(f"Monker manifest not found: {monker_manifest}")
    if sph_manifest and not sph_manifest.exists():
        print(f"⚠️ SPH manifest configured but missing: {sph_manifest}")
        sph_manifest = None

    # Vendor/lookup knobs
    ven = cfg.get("vendor", {}) or {}
    cache_dir        = ven.get("cache_dir", "data/vendor_cache")
    allow_pair_subs  = bool(ven.get("allow_pair_subs", True))
    max_stack_delta  = ven.get("max_stack_delta", None)
    max_stack_delta  = int(max_stack_delta) if max_stack_delta is not None else None

    # Label smoothing
    alpha = float((cfg.get("manifest_build", {}) or {}).get("alpha", 0.0))

    # 1) Expand scenario grid
    scenarios = _expand_scenarios_from_yaml(cfg)
    if scenarios.empty:
        raise RuntimeError("No scenarios expanded from manifest_build.scenarios in your YAML.")

    # 2) Instantiate lookup (Monker + optional SPH)
    lookup = PreflopRangeLookup(
        monker_manifest_parquet=str(monker_manifest),
        sph_manifest_parquet=(str(sph_manifest) if sph_manifest else None),
        s3_client=None,
        s3_vendor=None,
        cache_dir=cache_dir,
        allow_pair_subs=allow_pair_subs,
        max_stack_delta=max_stack_delta,
    )

    # 3) Query & materialize (with context backoff)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    backfilled = 0
    backoff_map = _load_ctx_backoff(cfg)

    for _, s in scenarios.iterrows():
        orig_ctx = str(s["ctx"])
        # build a backoff chain: [orig, backoff1, backoff2, ...] without cycles
        chain = []
        seen = set()
        c = orig_ctx
        while c and c not in seen:
            chain.append(c)
            seen.add(c)
            c = backoff_map.get(c)  # None ends the chain

        vec = None
        effective_ctx = orig_ctx
        used_backoff = False

        for trial_ctx in chain:
            vec = _lookup_vec169(
                lookup,
                stack_bb=int(s["stack_bb"]),
                hero_pos=str(s["hero_pos"]),
                opener_pos=str(s["opener_pos"]),
                ctx=str(trial_ctx),
            )
            if vec is not None:
                effective_ctx = trial_ctx
                used_backoff = (trial_ctx != orig_ctx)
                break

        if vec is None:
            skipped += 1
            continue

        if used_backoff:
            backfilled += 1

        v = np.asarray(vec, dtype=np.float32)
        if alpha > 0.0:
            v = v + (alpha / 169.0)
        v = (v / max(1e-12, float(v.sum()))).astype(np.float32)

        row = {k: s[k] for k in SCENARIO_KEYS}
        row["ctx_effective"] = effective_ctx
        row["used_ctx_backoff"] = bool(used_backoff)
        row["weight"] = 1.0
        for i, val in enumerate(v):
            row[f"y_{i}"] = float(val)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(
        f"✅ wrote {out_parquet} with {len(out):,} scenarios"
        f"{'' if not skipped else f' (skipped unresolved: {skipped})'}"
        f"{'' if not backfilled else f' (used ctx_backoff: {backfilled})'}"
    )
    print("   Columns:", ", ".join(SCENARIO_KEYS + ["ctx_effective", "used_ctx_backoff"]), ", y_0..y_168, weight")
    return out

# ====================== CLI ======================

def run_from_config(cfg_name_or_path: str) -> None:
    cfg = load_model_config(cfg_name_or_path)
    build_rangenet_preflop_from_cfg(cfg)

def main():
    ap = argparse.ArgumentParser("Build RangeNet Preflop parquet from YAML scenarios + Monker/SPH lookup")
    ap.add_argument("--config", type=str, default="rangenet/preflop",
                    help="Model name or YAML path")
    args = ap.parse_args()
    run_from_config(args.config)

if __name__ == "__main__":
    main()
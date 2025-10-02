from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.positions import canon_pos
from ml.config.solver_profiles import profile_for
from typing import List, Dict, Optional, Tuple
from ml.config.bet_menus import BET_SIZE_MENUS, DEFAULT_MENU
from ml.etl.rangenet.preflop.monker_manifest import expected_menu_id
from ml.range.solvers.utils.sanitize_pairs import sanitize_position_pairs
from infra.storage.s3_client import S3Client
from ml.features.boards.board_clusterers.utils import discover_representative_flops
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from ml.range.solvers.keying import s3_key_for_solve, solve_sha1

# =========================
# Size presets (deterministic)
# =========================
# Pick sane NL10 presets and keep them constant across training.
OPEN_X = {"UTG": 3.0, "HJ": 2.5, "CO": 2.5, "BTN": 2.5, "SB": 3.0}
THREEBET_X = {"IP": 7.5, "OOP": 9.0}  # final 3bet size
FOURBET_X = 24.0                      # final 4bet size

# Exact ids your BET_SIZE_MENUS exposes
MENU_TAG_TO_ID = {
    "srp_ip":    "srp_hu.PFR_IP",
    "srp_oop":   "srp_hu.Caller_OOP",  # OOP caller (donk after check line)
    "3bet_ip":   "3bet_hu.Aggressor_IP",
    "3bet_oop":  "3bet_hu.Aggressor_OOP",
    "4bet_ip":   "4bet_hu.Aggressor_IP",
    "4bet_oop":  "4bet_hu.Aggressor_OOP",
    "limp":      "limped_single.SB_IP",
    "limp_multi":"limped_multi.Any",
}

# =========================
# Helpers: ctx + topology
# =========================
def _ctx_for_lookup(ctx: str) -> str:
    """
    Route SRP-like contexts to 'SRP' since your Monker index is SRP-centric.
    Other ctx pass through (your lookup has built-in fallback).
    """
    c = str(ctx).upper()
    return "SRP" if c in ("VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL", "SRP") else c


def _infer_topology_and_roles(ctx: str, ip: str, oop: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Minimal deterministic inference from (ctx, ip, oop) → (topology, opener, three_bettor).
    We use common poker heuristics that are sufficient to bind menus + pot sizes:

      - SRP: if pairing is late-pos vs blind, opener=late-pos; else opener=ip by default.
      - 3bet: if three-bettor is OOP (common vs BTN), assume OOP; else IP.
      - 4bet: we don’t need opener/3bettor to compute pot, but we return reasonable defaults.
      - Limped: opener='LIMP', three_bettor=None.

    This is for *manifest* construction only; training sees consistent menus and pots.
    """
    c = str(ctx).upper()
    ip, oop = canon_pos(ip), canon_pos(oop)

    if c in ("SRP", "VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL"):
        topo = "srp_hu"
        # late-pos vs blind → late-pos opened
        late = {"BTN", "CO"}
        blinds = {"SB", "BB"}
        if ip in late and oop in blinds:
            opener = ip
        elif oop in late and ip in blinds:
            opener = oop
        else:
            # default: assume IP was the opener
            opener = ip
        return topo, opener, None

    if c == "VS_3BET":
        topo = "3bet_hu"
        # typical pool: blinds 3bet OOP vs BTN/CO IP
        if ip in {"BTN", "CO"} and oop in {"SB", "BB"}:
            opener = ip
            three_bettor = oop  # OOP 3-bettor
        elif oop in {"BTN", "CO"} and ip in {"SB", "BB"}:
            opener = oop
            three_bettor = ip   # OOP 3-bettor
        else:
            # default: three-bettor is the IP
            opener = oop
            three_bettor = ip
        return topo, opener, three_bettor

    if c == "VS_4BET":
        topo = "4bet_hu"
        # default placeholders; menus/pot do not require exact seats here
        # assume 4-bettor is OOP if BTN vs blind, otherwise IP
        if ip in {"BTN", "CO"} and oop in {"SB", "BB"}:
            opener = ip
            three_bettor = oop  # 3-bettor OOP, 4-bettor could be either; not needed for pot/menu
        else:
            opener = oop
            three_bettor = ip
        return topo, opener, three_bettor

    if c == "LIMPED_SINGLE":
        return "limped_single", "LIMP", None

    if c == "LIMPED_MULTI":
        # multiway is not used here (builder is HU), keep a safe label anyway
        return "limped_multi", "LIMP", None

    # default to SRP if unknown
    return "srp_hu", ip, None


# =========================
# Pots: deterministic formulas
# =========================
def _pot_srp(open_x: float) -> float:
    return 1.5 + 2.0 * float(open_x)


def _pot_3bet(final_x: float) -> float:
    return 1.5 + 2.0 * float(final_x)


def _pot_4bet(final_x: float) -> float:
    return 1.5 + 2.0 * float(final_x)


def _compute_pot_bb(ctx: str, opener: Optional[str], ip: str, three_bettor: Optional[str]) -> float:
    topo, opener_h, three_h = _infer_topology_and_roles(ctx, ip, "X")  # we only need the shape
    c = str(ctx).upper()
    if c in ("SRP", "VS_OPEN", "OPEN", "VS_OPEN_RFI", "BLIND_VS_STEAL"):
        op = opener or opener_h or "BTN"
        x = OPEN_X.get(op, 2.5)
        return _pot_srp(x)
    if c == "VS_3BET":
        # IP or OOP 3-bettor changes final size preset
        ip_is_three = (three_bettor == ip)
        final = THREEBET_X["IP"] if ip_is_three else THREEBET_X["OOP"]
        return _pot_3bet(final)
    if c == "VS_4BET":
        return _pot_4bet(FOURBET_X)
    if c == "LIMPED_SINGLE":
        return 1.5
    if c == "LIMPED_MULTI":
        return 1.5  # HU builder won’t use multiway; safe default
    return 6.0


# =========================
# Menu binding (topology+roles)
# =========================
def _menu_for(ctx, ip, oop, opener, three_bettor, menu_tag: str | None = None):
    topo, opener_h, three_h = _infer_topology_and_roles(ctx, ip, oop)

    # 1) explicit tag wins
    if menu_tag:
        tag = str(menu_tag).strip().lower()
        menu_id = MENU_TAG_TO_ID.get(tag)
        if not menu_id:
            raise ValueError(f"Unknown bet_menus tag '{menu_tag}'")
        if menu_id not in BET_SIZE_MENUS:
            raise ValueError(f"Menu id '{menu_id}' from tag '{menu_tag}' not in BET_SIZE_MENUS")
        return menu_id, BET_SIZE_MENUS[menu_id]

    # 2) fall back to inference (unchanged)
    menu_id = expected_menu_id(
        topo=topo,
        ip=canon_pos(ip),
        oop=canon_pos(oop),
        opener=(opener or opener_h),
        three_bettor=(three_bettor or three_h),
    )
    if not menu_id:
        if topo == "limped_multi": menu_id = "limped_multi.Any"
        elif topo == "limped_single": menu_id = "limped_single.SB_IP"
        else:
            raise ValueError(f"Cannot derive bet_sizing_id for ctx={ctx} topo={topo} ip={ip} oop={oop}")
    if menu_id not in BET_SIZE_MENUS:
        raise ValueError(f"Unknown bet_sizing_id '{menu_id}'")
    return menu_id, BET_SIZE_MENUS[menu_id]

# =========================
# Builder
# =========================
def build_manifest(cfg: dict) -> pd.DataFrame:
    mb = cfg.get("manifest_build", {}) or {}
    sv = cfg.get("solver", {}) or {}
    inputs = cfg.get("inputs", {}) or {}
    rake_tier = str(sv.get("rake_tier", "nl10_5pct_1bbcap"))

    # Scenarios (config-driven)
    scenarios = mb.get("scenarios") or []
    if not scenarios:
        scenarios = [{
            "name": "SRP",
            "ctx": "SRP",
            "stacks_bb": [100],
            "position_pairs": [("BTN", "BB")],
        }]

    n_clusters_limit = int(mb.get("board_clusters_limit", 24))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool = int(mb.get("sample_pool", 50000))
    seed = int(cfg.get("seed", 42))
    allow_pair_subs = bool(mb.get("allow_pair_subs", False))
    include_missing = bool(mb.get("include_missing", False))  # largely moot: lookup has built-in fallback
    raw_delta = mb.get("lookup_max_stack_delta")
    max_stack_delta: Optional[int]
    if raw_delta is None:
        max_stack_delta = None
    else:
        try:
            max_stack_delta = int(raw_delta)
        except (TypeError, ValueError):
            max_stack_delta = None

    # Preflop range lookup (Monker-only)
    lookup = PreflopRangeLookup(
        monker_manifest_parquet=inputs.get("monker_manifest", "data/artifacts/monker_manifest.parquet"),
        sph_manifest_parquet=inputs.get("sph_manifest", "data/artifacts/sph_manifest.parquet"),
        s3_client=S3Client(),
        s3_vendor=inputs.get("vendor_s3_prefix", "data/vendor"),
        cache_dir=sv.get("local_cache_dir", "data/vendor_cache"),
        allow_pair_subs=allow_pair_subs,
        max_stack_delta=max_stack_delta,
    )

    # Flop board sampling (deterministic)
    clusterer = load_board_clusterer(cfg)
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )
    cluster_ids_sorted = sorted(boards_by_cluster.keys(), key=int)

    rows: List[dict] = []
    total_jobs = kept = missing_rows = skipped = 0
    per_scenario_counts: Dict[str, Dict[str, int]] = {}

    for sc in scenarios:
        scenario_name = str(sc.get("name") or sc.get("ctx") or "SCENARIO").upper()
        ctx = str(sc.get("ctx") or scenario_name).upper()

        # Stacks + Pairs
        stacks = [float(x) for x in sc.get("stacks_bb", [100])]
        raw_pairs: List[Tuple[str, str]] = [(str(a), str(b)) for (a, b) in sc.get("position_pairs", [("BTN", "BB")])]
        ctx_up = str(ctx).upper()
        norm_ctx = {
            "LIMPED_SINGLE": "LIMPED_SINGLE",
            "LIMP_SINGLE": "LIMPED_SINGLE",
        }.get(ctx_up, ctx_up)

        pairs = sanitize_position_pairs(raw_pairs, ctx=norm_ctx)
        if not pairs:
            print(f"[warn] scenario {scenario_name} produced no legal (IP,OOP) pairs for ctx={ctx}")
            per_scenario_counts[scenario_name] = {"jobs": 0, "kept": 0, "missing": 0, "skipped": 0}
            continue

        planned = len(stacks) * len(pairs) * len(cluster_ids_sorted) * boards_per_cluster
        if planned > 200_000:
            print(f"[warn] {scenario_name}: ~{planned:,} jobs planned. Consider reducing boards/menus/stacks.")

        sc_jobs = sc_kept = sc_missing = sc_skipped = 0

        for stack in stacks:
            for (ip_pos, oop_pos) in pairs:
                ip_pos = canon_pos(ip_pos); oop_pos = canon_pos(oop_pos)
                if not ip_pos or not oop_pos or ip_pos == oop_pos:
                    continue

                # Roles + deterministic pot + menu
                topo, opener, three_bettor = _infer_topology_and_roles(ctx, ip_pos, oop_pos)
                pot_bb = _compute_pot_bb(ctx, opener, ip_pos, three_bettor)
                menu_id, bet_sizes = _menu_for(ctx, ip_pos, oop_pos, opener, three_bettor)

                # Resolve ranges (Monker-first + built-in fallback)
                try:
                    rng_ip, rng_oop, meta = lookup.ranges_for_pair(
                        stack_bb=stack, ip=ip_pos, oop=oop_pos, ctx=_ctx_for_lookup(ctx), strict=False
                    )
                except Exception as e:
                    rng_ip = rng_oop = None
                    meta = {"error": str(e)}

                if rng_ip is None or rng_oop is None:
                    # With built-in fallback this shouldn't happen; still respect include_missing
                    if not include_missing:
                        miss = len(cluster_ids_sorted) * boards_per_cluster
                        sc_skipped += miss; skipped += miss
                        continue

                for cluster_id in cluster_ids_sorted:
                    for board_tuple in boards_by_cluster[cluster_id]:
                        board_str = "".join(board_tuple)

                        # ⬇️ contextual solver knobs based on this menu/context
                        knobs = profile_for(menu_id)  # {"accuracy", "max_iter", "allin_threshold"}

                        params = {
                            "street": 1,
                            "scenario": scenario_name,
                            "ctx": ctx,
                            "topology": topo,
                            "rake_tier": rake_tier,

                            "positions": f"{ip_pos}v{oop_pos}",
                            "ip_actor_flop": ip_pos,
                            "oop_actor_flop": oop_pos,

                            "opener": opener,
                            "three_bettor": three_bettor,

                            "board_cluster_id": int(cluster_id),
                            "board": board_str,

                            "pot_bb": float(pot_bb),
                            "effective_stack_bb": float(stack),

                            "bet_sizing_id": menu_id,
                            "bet_sizes": bet_sizes,

                            # ranges (JSON-169 strings per your lookup)
                            "range_ip": rng_ip or "",
                            "range_oop": rng_oop or "",

                            # 🔧 solver knobs (contextual)
                            "accuracy": knobs["accuracy"],
                            "max_iter": knobs["max_iter"],
                            "allin_threshold": knobs["allin_threshold"],
                            "solver_version": "v1",  # keep whatever you already set earlier

                            # provenance
                            "range_ip_source_stack": meta.get("range_ip_source_stack"),
                            "range_oop_source_stack": meta.get("range_oop_source_stack"),
                            "range_ip_stack_delta": meta.get("range_ip_stack_delta"),
                            "range_oop_stack_delta": meta.get("range_oop_stack_delta"),
                            "range_ip_fallback_level": meta.get("range_ip_fallback_level"),
                            "range_oop_fallback_level": meta.get("range_oop_fallback_level"),
                            "range_pair_substituted": bool(meta.get("range_pair_substituted", False)),
                            "range_ip_source_pair": meta.get("range_ip_source_pair"),
                            "range_oop_source_pair": meta.get("range_oop_source_pair"),
                            "range_source": meta.get("source", "unknown"),
                        }

                        sha = solve_sha1(params)
                        s3_key = s3_key_for_solve(
                            params,
                            sha1=sha,
                            prefix="solver/outputs/v1"
                        )

                        rows.append({
                            **params,
                            "sha1": sha,
                            "s3_key": s3_key,
                            "node_key": "root",
                            "weight": 1.0,
                        })
                        total_jobs += 1
                        sc_jobs += 1
                        if not rng_ip or not rng_oop:
                            sc_missing += 1
                            missing_rows += 1
                        else:
                            sc_kept += 1
                            kept += 1

        per_scenario_counts[scenario_name] = {
            "jobs": sc_jobs, "kept": sc_kept, "missing": sc_missing, "skipped": sc_skipped
        }

    df = pd.DataFrame(rows)

    # --- DEDUPE on job identity ---
    before = len(df)
    # sha1 is derived from params (board, positions, ctx, stack, menu, etc.)
    if "sha1" in df.columns:
        df = df.drop_duplicates(subset=["sha1"]).reset_index(drop=True)
    else:
        # very defensive fallback: dedupe on a minimal identifying subset
        key_cols = [
            "street", "ctx", "topology",
            "ip_actor_flop", "oop_actor_flop",
            "effective_stack_bb", "pot_bb",
            "bet_sizing_id", "board", "board_cluster_id",
            "range_ip", "range_oop"
        ]
        key_cols = [c for c in key_cols if c in df.columns]
        df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    removed = before - len(df)

    # summary
    print(f"[dbg] scenarios={len(scenarios)} → jobs={total_jobs:,} "
          f"(kept={kept:,}, missing_rows={missing_rows:,}, skipped={skipped:,})")
    print(f"      deduped: removed {removed:,} duplicate jobs; final={len(df):,}")
    if per_scenario_counts:
        print("   per-scenario:")
        for name in sorted(per_scenario_counts.keys()):
            c = per_scenario_counts[name]
            print(f"     - {name}: jobs={c['jobs']:,} kept={c['kept']:,} "
                  f"missing={c['missing']:,} skipped={c['skipped']:,}")

    # quick peek
    if not df.empty:
        cols = ["ctx","topology","ip_actor_flop","oop_actor_flop","effective_stack_bb","pot_bb","bet_sizing_id","range_source"]
        print(df[cols].head(12).to_string(index=False))

    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build flop-only manifest for RangeNet Postflop (Monker-first).")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="model[/variant]/profile in your config loader")
    ap.add_argument("--out", type=str,
                    default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--scenario", type=str, default=None,
                    help="Optional: only build a single scenario name (case-insensitive)")
    args = ap.parse_args()

    cfg = load_model_config(model=args.config)

    if args.scenario:
        mb = dict(cfg.get("manifest_build", {}) or {})
        scenarios = [s for s in (mb.get("scenarios") or []) if str(s.get("name","")).upper() == args.scenario.upper()]
        if not scenarios:
            print(f"[warn] no scenario named '{args.scenario}' found; building all.")
        else:
            mb["scenarios"] = scenarios
            cfg = {**cfg, "manifest_build": mb}

    print("config loaded")
    df = build_manifest(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote FLOP manifest: {out} rows={len(df):,}")


if __name__ == "__main__":
    main()
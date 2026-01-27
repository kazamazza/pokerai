# ml/etl/rangenet/postflop/audit_preflop_range_coverage.py
# ==========================================================
# Audit: PreflopRangeLookup coverage for planned postflop scenarios.
# - NO imports from old helper modules.
# - Only depends on PreflopRangeLookup (+ S3Client) which you trust.
#
# It checks: for each (scenario ctx, stack_bb, ip_pos, oop_pos) cell:
#   can lookup.ranges_for_pair(...) return both ranges?
#
# PASS/FAIL exit code:
#   0 = pass, 2 = fail
#
# Usage:
#   python -m ml.etl.rangenet.postflop.audit_preflop_range_coverage \
#     --postflop-config ml/config/rangenet/postflop_root/base.yaml \
#     --solver-yaml ml/config/solver.yaml \
#     --stake NL10 \
#     --out reports/coverage/preflop_range_coverage_NL10.json \
#     --fail-below 0.98
# ==========================================================

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


from infra.storage.s3_client import S3Client
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup


# -----------------------------
# Canonical poker seat labels (v1 HU only)
# -----------------------------
POS_SET = {"UTG", "MP", "CO", "BTN", "SB", "BB"}


def canon_pos(p: str) -> str:
    s = (p or "").strip().upper()
    if s == "HJ":
        return "MP"  # optional: if your project treats HJ as MP
    return s


def canon_pair(a: str, b: str) -> Tuple[str, str]:
    return canon_pos(a), canon_pos(b)


# -----------------------------
# Context mapping for range lookup (explicit + local)
# -----------------------------
def ctx_for_range_lookup(ctx: str) -> str:
    """
    Single source of truth inside THIS file.
    If your vendor range index is SRP-centric, map VS_OPEN -> SRP here.
    """
    c = (ctx or "").strip().upper()
    if c in {"VS_OPEN", "OPEN", "RFI"}:
        return "SRP"
    return c


# -----------------------------
# Scenario legality (explicit + local)
# -----------------------------
def is_legal_pair_for_ctx(ctx: str, ip: str, oop: str) -> bool:
    """
    Minimal HU legality filters to prevent nonsense pairs.
    This is not trying to be clever—just stops bad inputs.
    """
    c = (ctx or "").strip().upper()
    ip, oop = canon_pos(ip), canon_pos(oop)

    if ip == oop:
        return False
    if ip not in POS_SET or oop not in POS_SET:
        return False

    # Limped single in your scenarios is BB vs SB only.
    if c == "LIMPED_SINGLE":
        return (ip, oop) == ("BB", "SB")

    # Otherwise allow the pairs specified in YAML; no extra restrictions here.
    return True


def read_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_stacks_for_ctx(solver_yaml: Dict[str, Any], *, stake: str, ctx: str) -> List[float]:
    """
    Reads stacks_by_ctx from solver.yaml shaped like:

      Stakes.NL10:
        stacks_by_ctx:
          VS_OPEN: [25, 60, 100, 150]
          ...

    i.e. top-level key is literally "Stakes.NL10".
    """
    stake_key = str(stake or "").strip().upper()
    top_key = f"Stakes.{stake_key}"

    stake_cfg = solver_yaml.get(top_key)
    if not isinstance(stake_cfg, dict):
        raise ValueError(f"solver.yaml missing top-level '{top_key}'")

    stacks_by_ctx = stake_cfg.get("stacks_by_ctx")
    if not isinstance(stacks_by_ctx, dict):
        raise ValueError(f"solver.yaml '{top_key}.stacks_by_ctx' missing or invalid")

    c = (ctx or "").strip().upper()
    stacks = stacks_by_ctx.get(c)
    if stacks is None:
        raise ValueError(f"solver.yaml has no stacks_by_ctx entry for ctx={c} under {top_key}")

    return [float(x) for x in stacks]

@dataclass(frozen=True)
class Cell:
    scenario: str
    ctx: str
    ctx_lookup: str
    stack_bb: float
    ip_pos: str
    oop_pos: str
    ok: bool
    meta: Dict[str, Any]


def audit(
    *,
    postflop_cfg_path: str,
    solver_yaml_path: str,
    stake: str,
    fail_below: float,
) -> Tuple[bool, Dict[str, Any]]:
    cfg = read_yaml(postflop_cfg_path)
    solver_yaml = read_yaml(solver_yaml_path)

    mb = cfg.get("manifest_build") or {}
    scenarios = mb.get("scenarios") or []
    if not scenarios:
        raise ValueError("postflop config missing manifest_build.scenarios")

    inputs = cfg.get("inputs") or {}
    solver = cfg.get("solver") or {}

    allow_pair_subs = bool(mb.get("allow_pair_subs", True))
    max_stack_delta = mb.get("lookup_max_stack_delta", 200)
    max_stack_delta = int(max_stack_delta) if max_stack_delta is not None else None

    lookup = PreflopRangeLookup(
        monker_manifest_parquet=inputs.get("monker_manifest", "data/artifacts/monker_manifest.parquet"),
        sph_manifest_parquet=inputs.get("sph_manifest", "data/artifacts/sph_manifest.parquet"),
        s3_client=S3Client(),
        s3_vendor=inputs.get("vendor_s3_prefix", "data/vendor"),
        cache_dir=solver.get("local_cache_dir", "data/vendor_cache"),
        allow_pair_subs=allow_pair_subs,
        max_stack_delta=max_stack_delta,
    )

    cells: List[Cell] = []
    totals = {"cells": 0, "ok": 0, "missing": 0, "errors": 0}

    missing_by_ctx: Dict[str, int] = {}
    missing_by_pair: Dict[str, int] = {}
    errors_by_reason: Dict[str, int] = {}

    for sc in scenarios:
        sc_name = str(sc.get("name") or "SCENARIO").upper()
        ctx = str(sc.get("ctx") or "").strip().upper()
        if not ctx:
            raise ValueError(f"scenario {sc_name} missing ctx")

        ctx_lu = ctx_for_range_lookup(ctx)
        stacks = get_stacks_for_ctx(solver_yaml, stake=stake, ctx=ctx)

        raw_pairs = sc.get("position_pairs") or []
        if not raw_pairs:
            raise ValueError(f"scenario {sc_name} has no position_pairs")

        # Use EXACT pairs from YAML, just canonicalize; no external sanitize.
        pairs: List[Tuple[str, str]] = []
        for a, b in raw_pairs:
            ip, oop = canon_pair(str(a), str(b))
            if not is_legal_pair_for_ctx(ctx, ip, oop):
                continue
            pairs.append((ip, oop))

        for stack_bb in stacks:
            for (ip, oop) in pairs:
                totals["cells"] += 1

                meta: Dict[str, Any] = {}
                ok = False
                try:
                    rng_ip, rng_oop, rmeta = lookup.ranges_for_pair(
                        stack_bb=float(stack_bb),
                        ip=ip,
                        oop=oop,
                        ctx=ctx_lu,
                        strict=False,
                    )
                    meta.update(rmeta or {})
                    ok = (rng_ip is not None) and (rng_oop is not None)
                except Exception as e:
                    meta["error"] = str(e)
                    reason = type(e).__name__
                    errors_by_reason[reason] = errors_by_reason.get(reason, 0) + 1
                    totals["errors"] += 1
                    ok = False

                if ok:
                    totals["ok"] += 1
                else:
                    totals["missing"] += 1
                    missing_by_ctx[ctx] = missing_by_ctx.get(ctx, 0) + 1
                    pair_key = f"{ip}v{oop}"
                    missing_by_pair[pair_key] = missing_by_pair.get(pair_key, 0) + 1

                cells.append(
                    Cell(
                        scenario=sc_name,
                        ctx=ctx,
                        ctx_lookup=ctx_lu,
                        stack_bb=float(stack_bb),
                        ip_pos=ip,
                        oop_pos=oop,
                        ok=ok,
                        meta=meta,
                    )
                )

    coverage = (totals["ok"] / max(1, totals["cells"]))
    passed = coverage >= float(fail_below)

    report = {
        "passed": passed,
        "fail_below": float(fail_below),
        "coverage": coverage,
        "totals": totals,
        "missing_by_ctx": dict(sorted(missing_by_ctx.items(), key=lambda x: (-x[1], x[0]))),
        "missing_by_pair": dict(sorted(missing_by_pair.items(), key=lambda x: (-x[1], x[0]))),
        "errors_by_reason": dict(sorted(errors_by_reason.items(), key=lambda x: (-x[1], x[0]))),
        "cells": [asdict(c) for c in cells],
    }
    return passed, report


def _print_summary(report: Dict[str, Any]) -> None:
    print("=== Preflop Range Coverage Audit ===")
    print(f"passed:   {report['passed']}")
    print(f"coverage: {report['coverage']:.4f} (threshold={report['fail_below']:.4f})")
    t = report["totals"]
    print(f"cells:    {t['cells']}  ok: {t['ok']}  missing: {t['missing']}  errors: {t['errors']}")
    if report["missing_by_ctx"]:
        print("missing_by_ctx (top):")
        for k, v in list(report["missing_by_ctx"].items())[:10]:
            print(f"  - {k}: {v}")
    if report["missing_by_pair"]:
        print("missing_by_pair (top):")
        for k, v in list(report["missing_by_pair"].items())[:10]:
            print(f"  - {k}: {v}")
    if report["errors_by_reason"]:
        print("errors_by_reason (top):")
        for k, v in list(report["errors_by_reason"].items())[:10]:
            print(f"  - {k}: {v}")


def cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--postflop-config", type=str, required=True, help="Path to postflop YAML containing manifest_build.scenarios")
    ap.add_argument("--solver-yaml", type=str, default="ml/config/solver.yaml", help="Path to solver.yaml (stakes config)")
    ap.add_argument("--stake", type=str, default="NL10", help="Stake key inside solver.yaml (e.g. NL10)")
    ap.add_argument("--out", type=str, default="reports/coverage/preflop_range_coverage.json", help="Output JSON report path")
    ap.add_argument("--fail-below", type=float, default=0.98, help="Fail if coverage < this threshold")
    args = ap.parse_args()

    passed, report = audit(
        postflop_cfg_path=args.postflop_config,
        solver_yaml_path=args.solver_yaml,
        stake=args.stake,
        fail_below=args.fail_below,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"wrote: {out}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(cli())
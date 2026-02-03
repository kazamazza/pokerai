from __future__ import annotations

import sys
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.policy.extractor_invariants import validate_extractor_output
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
import argparse
import gzip
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, cast

import boto3


# ----------------------------
# Minimal ctx + root-kind logic
# ----------------------------

def ctx_from_bet_sizing_id(bet_sizing_id: str) -> str:
    s = (bet_sizing_id or "").strip()
    if s.startswith("srp_hu."):
        return "VS_OPEN"
    if s == "bvs.Any":
        return "BLIND_VS_STEAL"
    if s.startswith("3bet_hu."):
        return "VS_3BET"
    if s.startswith("4bet_hu."):
        return "VS_4BET"
    if s.startswith("limped_single."):
        return "LIMPED_SINGLE"
    if s.startswith("limped_multi."):
        return "LIMPED_MULTI"
    return ""


OOP_ROOT_KIND = Literal["donk", "bet"]

_BET_SIZING_ID_TO_OOP_ROOT_KIND: Dict[str, OOP_ROOT_KIND] = {
    "srp_hu.PFR_IP": "donk",
    "srp_hu.Caller_OOP": "donk",
    "bvs.Any": "donk",
    "3bet_hu.Aggressor_IP": "donk",
    "3bet_hu.Aggressor_OOP": "bet",
    "4bet_hu.Aggressor_IP": "donk",
    "4bet_hu.Aggressor_OOP": "bet",
    "limped_single.BB_IP": "donk",   # <-- IMPORTANT: donk (matches worker)
    "limped_multi.Any": "donk",      # <-- IMPORTANT: donk (matches worker)
}

def oop_root_kind_for_bet_sizing_id(bet_sizing_id: str) -> OOP_ROOT_KIND:
    return _BET_SIZING_ID_TO_OOP_ROOT_KIND.get((bet_sizing_id or "").strip(), "donk")


# ----------------------------
# YAML stake params (raise_mults)
# ----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}

def load_raise_mults(solver_yaml: Dict[str, Any], stake_key: str) -> List[float]:
    st = solver_yaml.get(stake_key) or {}
    rm = st.get("raise_mult") or []
    if not isinstance(rm, list) or not rm:
        raise ValueError(f"{stake_key}.raise_mult missing/invalid in solver.yaml")
    return [float(x) for x in rm]


# ----------------------------
# S3 helpers
# ----------------------------

def s3_client(region: str):
    return boto3.client("s3", region_name=region)

def list_solver_keys(s3, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for it in page.get("Contents") or []:
            k = it.get("Key")
            if isinstance(k, str) and k.endswith("/output_result.json.gz"):
                keys.append(k)
    return keys

def download_if_missing(s3, bucket: str, key: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return
    s3.download_file(bucket, key, str(dst))
    if not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded file missing/empty: {dst}")


# ----------------------------
# Parse fields from your key schema
# Example:
# solver/outputs/v1/street=1/pos=UTGvBB/stack=25/pot=6.5/board=AsJs5c/acc=0.02/sizes=srp_hu.PFR_IP/bf/<sha1>/size=25p/output_result.json.gz
# ----------------------------

_KEY_RE = re.compile(
    r"""
    ^.*?
    /street=(?P<street>\d+)
    /pos=(?P<pos>[A-Za-z]+)v(?P<opp>[A-Za-z]+)
    /stack=(?P<stack>[\d.]+)
    /pot=(?P<pot>[\d.]+)
    /board=(?P<board>[0-9TJQKAcdhs]{6,10})
    /acc=(?P<acc>[\d.]+)
    /sizes=(?P<sizes>[^/]+)
    /(?P<prefix2>[0-9a-f]{2})
    /(?P<sha1>[0-9a-f]{40})
    /size=(?P<sizepct>\d+)p
    /output_result\.json\.gz$
    """,
    re.VERBOSE,
)

@dataclass(frozen=True)
class ParsedKey:
    key: str
    street: int
    ip_pos: str
    oop_pos: str
    stack_bb: float
    pot_bb: float
    board: str
    accuracy: float
    bet_sizing_id: str
    sha1: str
    size_pct: int

def parse_key(key: str) -> ParsedKey:
    m = _KEY_RE.match(key)
    if not m:
        raise ValueError(f"Key did not match expected schema: {key}")

    street = int(m.group("street"))
    ip_pos = m.group("pos")
    oop_pos = m.group("opp")
    stack_bb = float(m.group("stack"))
    pot_bb = float(m.group("pot"))
    board = m.group("board")
    accuracy = float(m.group("acc"))
    bet_sizing_id = m.group("sizes")
    sha1 = m.group("sha1")
    size_pct = int(m.group("sizepct"))

    return ParsedKey(
        key=key,
        street=street,
        ip_pos=ip_pos,
        oop_pos=oop_pos,
        stack_bb=stack_bb,
        pot_bb=pot_bb,
        board=board,
        accuracy=accuracy,
        bet_sizing_id=bet_sizing_id,
        sha1=sha1,
        size_pct=size_pct,
    )


def main() -> None:
    ap = argparse.ArgumentParser("Sanity check: read solves from S3 bucket and run extractor+invariants")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
    ap.add_argument("--prefix", default="solver/outputs/v1/", help="S3 prefix to scan")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of solve files checked (0 = all)")
    ap.add_argument("--cache-dir", default="data/cache/solves", help="Local cache root")
    ap.add_argument("--strict", choices=["skip", "fail"], default="skip")

    ap.add_argument("--solver-yaml", required=True, help="path to solver.yaml")
    ap.add_argument("--stake-key", default="Stakes.NL10", help="e.g. Stakes.NL10")

    ap.add_argument("--debug-jsonl", default="", help="Write failures to jsonl (optional)")

    args = ap.parse_args()

    s3 = s3_client(args.region)
    keys = list_solver_keys(s3, args.bucket, args.prefix)
    keys.sort()

    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    if not keys:
        print("⚠️ No solver outputs found under prefix:", args.prefix)
        return

    solver_yaml = load_yaml(args.solver_yaml)
    raise_mults = load_raise_mults(solver_yaml, args.stake_key)

    extractor = TexasSolverExtractor()

    cache_root = Path(args.cache_dir).resolve()
    dbg_fh = open(args.debug_jsonl, "a", encoding="utf-8") if args.debug_jsonl else None

    total = 0
    ok = 0
    failed = 0
    reasons: Dict[str, int] = {}

    def fail(reason: str, payload: Dict[str, Any]) -> None:
        nonlocal failed
        failed += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        if dbg_fh:
            dbg_fh.write(json.dumps({"reason": reason, **payload}) + "\n")
        if args.strict == "fail":
            raise RuntimeError(f"{reason}: {payload.get('key')}")

    for key in keys:
        total += 1
        ex = None
        try:
            pk = parse_key(key)
            local_path = (cache_root / pk.key).resolve()
            download_if_missing(s3, args.bucket, pk.key, local_path)

            ctx = ctx_from_bet_sizing_id(pk.bet_sizing_id)
            root_kind = oop_root_kind_for_bet_sizing_id(pk.bet_sizing_id)

            ex = extractor.extract(
                str(local_path),
                ctx=ctx,
                ip_pos=pk.ip_pos,
                oop_pos=pk.oop_pos,
                board=pk.board,
                pot_bb=pk.pot_bb,
                stack_bb=pk.stack_bb,
                bet_sizing_id=pk.bet_sizing_id,
                size_pct=int(pk.size_pct),
                root_actor="oop",
                root_bet_kind=root_kind,
                raise_mults=raise_mults,
            )

            validate_extractor_output(ex)
            ok += 1

        except Exception as e:
            payload = {
                "key": pk.key if "pk" in locals() else key,
                "err": str(e),
            }

            # 🔥 THIS is what you need to see
            if ex is not None:
                payload.update({
                    "ex_ok": getattr(ex, "ok", None),
                    "ex_reason": getattr(ex, "reason", None),
                    "meta": getattr(ex, "meta", None),
                    "root_mix": getattr(ex, "root_mix", None),
                    "facing_mix": getattr(ex, "facing_mix", None),
                })

            fail("extract_or_validate_failed", payload)

    if dbg_fh:
        dbg_fh.close()

    print("\n====================")
    print("SANITY SUMMARY (bucket)")
    print("====================")
    print(f"bucket        : {args.bucket}")
    print(f"prefix        : {args.prefix}")
    print(f"total checked : {total}")
    print(f"ok            : {ok}")
    print(f"failed        : {failed}")

    if reasons:
        print("\nFailures by reason:")
        for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
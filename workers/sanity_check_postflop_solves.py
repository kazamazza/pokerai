from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

from ml.etl.rangenet.postflop.build_postflop_policy import size_key_for
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
from ml.policy.extractor_invariants import validate_extractor_output
from ml.policy.solver_action_mapping import oop_root_kind_for_bet_sizing_id
from workers.solver_job_schema import parse_bet_sizes

# -------------------------
# S3 helpers
# -------------------------
_S3CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=90,
    tcp_keepalive=True,
)

def s3_client(region: str):
    return boto3.client("s3", region_name=region, config=_S3CFG)

def s3_list_keys(s3, bucket: str, prefix: str) -> Iterable[str]:
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj.get("Key")
            if key:
                yield key
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")

def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        http = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code in ("404", "NotFound") or http == 404:
            return False
        raise

def s3_download_to(s3, bucket: str, key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))

# -------------------------
# Manifest parsing
# -------------------------
def parse_bet_sizes_cell(cell) -> List[int]:
    """
    Match your submitter/policy builder behavior.
    Accepts [0.33, 0.66] or [33, 66] etc. Returns integer percents.
    """
    if cell is None:
        return []

    # unwrap pyarrow
    try:
        if hasattr(cell, "as_py"):
            cell = cell.as_py()
    except Exception:
        pass

    seq = cell if isinstance(cell, list) else [cell]
    out: List[int] = []
    seen = set()

    for it in seq:
        if it is None:
            continue
        v = it.get("element") if isinstance(it, dict) else it
        try:
            f = float(v)
        except Exception:
            continue
        pct = int(round(f * 100)) if f <= 3.0 else int(round(f))
        if 1 <= pct <= 200 and pct not in seen:
            out.append(pct)
            seen.add(pct)

    return out

# -------------------------
# Reporting
# -------------------------
@dataclass
class CheckResult:
    ok: bool
    reason: str
    key: str
    sha1: Optional[str] = None
    street: Optional[int] = None
    bet_sizing_id: Optional[str] = None
    size_pct: Optional[int] = None

def _summarize(results: List[CheckResult]) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    bad = total - ok
    print("\n====================")
    print("SANITY SUMMARY")
    print("====================")
    print(f"total checked : {total}")
    print(f"ok            : {ok}")
    print(f"failed        : {bad}")

    if bad:
        by_reason: Dict[str, int] = {}
        for r in results:
            if r.ok:
                continue
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1

        print("\nFailures by reason:")
        for k, v in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k:30s} {v}")

        print("\nSample failures:")
        shown = 0
        for r in results:
            if r.ok:
                continue
            print(f"  - reason={r.reason} key={r.key} sha1={r.sha1} size={r.size_pct} sizing={r.bet_sizing_id}")
            shown += 1
            if shown >= 10:
                break

# -------------------------
# Core check logic
# -------------------------
def sanity_check_key(
    *,
    extractor,
    local_path: Path,
    ctx: str,
    ip_pos: str,
    oop_pos: str,
    board: str,
    pot_bb: float,
    stack_bb: float,
    bet_sizing_id: str,
    size_pct: int,
    raise_mults: List[float],
) -> None:
    """
    This should mirror your policy builder extractor call as closely as possible.
    """
    root_kind = oop_root_kind_for_bet_sizing_id(bet_sizing_id)

    ex = extractor.extract(
        str(local_path),
        ctx=ctx,
        ip_pos=ip_pos,
        oop_pos=oop_pos,
        board=board,
        pot_bb=pot_bb,
        stack_bb=stack_bb,
        bet_sizing_id=bet_sizing_id,
        size_pct=int(size_pct),
        root_actor="oop",
        root_bet_kind=root_kind,
        raise_mults=raise_mults,
    )

    # Non-negotiable invariants
    validate_extractor_output(ex)

def main() -> None:
    ap = argparse.ArgumentParser("Sanity check: S3 solves -> extractor -> invariants")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
    ap.add_argument("--prefix", default="solver/outputs/v1/", help="S3 prefix for solved outputs")
    ap.add_argument("--cache-dir", default="data/cache/solver_outputs")
    ap.add_argument("--manifest", default=None, help="Optional manifest parquet (recommended)")
    ap.add_argument("--limit-rows", type=int, default=0, help="If manifest provided, limit manifest rows")
    ap.add_argument("--limit-keys", type=int, default=0, help="If S3-only mode, limit number of keys")
    ap.add_argument("--strict", choices=["fail", "skip"], default="skip")
    ap.add_argument("--raise-mults", default=None, help="Comma list like '2.0,3.0,4.0' (fallback if not in your code)")
    args = ap.parse_args()

    # Instantiate your real extractor
    if TexasSolverExtractor is None:
        raise SystemExit(
            "Replace the placeholder imports at top with your real TexasSolverExtractor / validate / vocabs / helpers."
        )

    extractor = TexasSolverExtractor()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    s3 = s3_client(args.region)

    # Raise mults: ideally you load from solver.yaml like worker does
    if args.raise_mults:
        raise_mults = [float(x.strip()) for x in args.raise_mults.split(",") if x.strip()]
    else:
        # if you have a canonical source inside code, set it here or load YAML
        raise_mults = [2.0, 3.0, 4.0]

    results: List[CheckResult] = []

    # -------------------------
    # Mode A: Manifest-driven (best)
    # -------------------------
    if args.manifest:
        df = pd.read_parquet(args.manifest)
        if args.limit_rows and args.limit_rows > 0:
            df = df.head(args.limit_rows)

        for r in tqdm(df.itertuples(index=False), total=len(df), desc="sanity(manifest)"):
            base_s3_key = str(getattr(r, "s3_key"))
            sha1 = str(getattr(r, "sha1"))
            street = int(getattr(r, "street"))
            board = str(getattr(r, "board") or "")
            ip_pos = str(getattr(r, "ip_pos") or "")
            oop_pos = str(getattr(r, "oop_pos") or "")
            ctx = str(getattr(r, "ctx") or "")
            bet_sizing_id = str(getattr(r, "bet_sizing_id") or "")
            pot_bb = float(getattr(r, "pot_bb") or 0.0)
            stack_bb = float(getattr(r, "effective_stack_bb") or 0.0)

            if pot_bb <= 0 or stack_bb <= 0:
                results.append(CheckResult(False, "bad_pot_or_stack", base_s3_key, sha1, street, bet_sizing_id, None))
                if args.strict == "fail":
                    break
                continue

            sizes_pct = parse_bet_sizes(getattr(r, "bet_sizes", None))
            if not sizes_pct:
                results.append(CheckResult(False, "no_bet_sizes", base_s3_key, sha1, street, bet_sizing_id, None))
                if args.strict == "fail":
                    break
                continue

            for size_pct in sizes_pct:
                key = size_key_for(base_s3_key, int(size_pct))
                local_path = (cache_dir / key).resolve()

                try:
                    if not local_path.is_file():
                        if not s3_exists(s3, args.bucket, key):
                            results.append(CheckResult(False, "missing_solver_output_s3", key, sha1, street, bet_sizing_id, int(size_pct)))
                            if args.strict == "fail":
                                raise RuntimeError(f"missing s3 key={key}")
                            continue
                        s3_download_to(s3, args.bucket, key, local_path)

                    sanity_check_key(
                        extractor=extractor,
                        local_path=local_path,
                        ctx=ctx,
                        ip_pos=ip_pos,
                        oop_pos=oop_pos,
                        board=board,
                        pot_bb=pot_bb,
                        stack_bb=stack_bb,
                        bet_sizing_id=bet_sizing_id,
                        size_pct=int(size_pct),
                        raise_mults=raise_mults,
                    )
                    results.append(CheckResult(True, "ok", key, sha1, street, bet_sizing_id, int(size_pct)))
                except Exception as e:
                    results.append(CheckResult(False, f"extract_fail:{type(e).__name__}", key, sha1, street, bet_sizing_id, int(size_pct)))
                    if args.strict == "fail":
                        raise

        _summarize(results)
        return

    # -------------------------
    # Mode B: S3-only (useful if you don’t trust manifest yet)
    # -------------------------
    keys = [k for k in s3_list_keys(s3, args.bucket, args.prefix) if k.endswith("output_result.json.gz")]
    if args.limit_keys and args.limit_keys > 0:
        keys = keys[: args.limit_keys]

    print(f"Found {len(keys)} solve keys under s3://{args.bucket}/{args.prefix}")

    # In S3-only mode we *can’t* pass ctx/ip/oop/board/pot/stack reliably unless you encode them in the key,
    # so this mode is only good for verifying “file exists + readable + extractor can parse structure”
    # if your extractor supports that. If not, keep using manifest-driven mode.
    for key in tqdm(keys, desc="sanity(s3-only)"):
        local_path = (cache_dir / key).resolve()
        try:
            if not local_path.is_file():
                s3_download_to(s3, args.bucket, key, local_path)
            # If your extractor needs full context, you cannot fully validate in this mode.
            # So we just mark as ok if it’s readable / non-empty:
            if local_path.stat().st_size <= 0:
                raise RuntimeError("empty file")
            results.append(CheckResult(True, "ok_file", key))
        except Exception as e:
            results.append(CheckResult(False, f"download_or_read_fail:{type(e).__name__}", key))
            if args.strict == "fail":
                raise

    _summarize(results)

if __name__ == "__main__":
    main()
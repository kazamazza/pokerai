from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

load_dotenv()

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ml.range.solvers.command_text import build_command_text
from ml.etl.utils.monker_range_converter import to_monker  # keep (vendor tolerance)
from infra.utils.gzip_file import gzip_file               # keep (trusted small util)


from typing import Dict, List, Literal, Optional, Union, TypedDict

Street = Literal["flop", "turn", "river"]
Role = Literal["ip", "oop"]
Kind = Literal["donk", "bet", "raise", "allin"]

BetKindMap = Dict[str, Union[List[float], List[int], bool]]   # keep compatible with build_command_text
BetSizesType = Dict[Street, Dict[Role, BetKindMap]]
# =========================================================
# YAML loader (no legacy config helpers)
# =========================================================
def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


# =========================================================
# S3 client (verified upload)
# =========================================================
_TRANSIENT_HTTP = {429, 500, 502, 503, 504}
_TRANSIENT_CODES = {
    "SlowDown", "Throttling", "ThrottlingException",
    "RequestTimeout", "RequestTimeoutException",
    "InternalError", "ServiceUnavailable",
}

_S3CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=90,
    tcp_keepalive=True,
)

def _s3_client(region: str):
    return boto3.client("s3", region_name=region, config=_S3CFG)

def _exists_in_s3(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code"))
        http = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code in ("404", "NotFound") or http == 404:
            return False
        raise

def _upload_file_with_verify(
    s3, local_path: Path, bucket: str, key: str,
    *, content_type: str = "application/json", content_encoding: str = "gzip",
    max_tries: int = 7,
) -> None:
    local_path = Path(local_path)
    size = local_path.stat().st_size

    backoff = 0.5
    for attempt in range(1, max_tries + 1):
        try:
            s3.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type, "ContentEncoding": content_encoding},
            )
            head = s3.head_object(Bucket=bucket, Key=key)
            remote = int(head.get("ContentLength", -1))
            if remote == size or size <= 0:
                return
            print(f"[warn] S3 size mismatch local={size} remote={remote}; retrying…")
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code"))
            http = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            transient = (http in _TRANSIENT_HTTP) or (code in _TRANSIENT_CODES)
            if (not transient) or (attempt == max_tries):
                raise
        time.sleep(backoff + random.random() * 0.3)
        backoff = min(10.0, backoff * 2.0)

    raise RuntimeError("S3 upload failed after retries/verify")


# =========================================================
# Solver menu root kind (NO PARSING, explicit mapping)
# =========================================================
OOP_ROOT_KIND = Literal["donk", "bet"]

# Only applies to NON-LIMP postflop trees.
_BET_SIZING_ID_TO_OOP_ROOT_KIND: Dict[str, OOP_ROOT_KIND] = {
    # SRP
    "srp_hu.PFR_IP": "donk",         # IP was aggressor -> OOP is caller
    "srp_hu.Caller_OOP": "donk",

    # BVS
    "bvs.Any": "donk",

    # 3bet
    "3bet_hu.Aggressor_IP": "donk",
    "3bet_hu.Aggressor_OOP": "bet",

    # 4bet
    "4bet_hu.Aggressor_IP": "donk",
    "4bet_hu.Aggressor_OOP": "bet",

    # NOTE: limps intentionally excluded
}

def is_limp_bet_sizing_id(bet_sizing_id: str) -> bool:
    k = (bet_sizing_id or "").strip()
    return k.startswith("limped_single") or k.startswith("limped_multi")

def _oop_root_kind(bet_sizing_id: str) -> OOP_ROOT_KIND:
    k = (bet_sizing_id or "").strip()

    # If you still want limp treated specially elsewhere, just return a default here.
    if k.startswith("limped_single") or k.startswith("limped_multi"):
        return "donk"   # or "bet", but doesn't matter if you override menus for limps

    return _BET_SIZING_ID_TO_OOP_ROOT_KIND.get(k, "donk")


# =========================================================
# Street naming for solver command_text
# =========================================================
StreetName = Literal["flop", "turn", "river"]

def _street_name(street: int) -> StreetName:
    if street == 1: return "flop"
    if street == 2: return "turn"
    if street == 3: return "river"
    raise ValueError(f"illegal street={street}")


# =========================================================
# Stake config reads from solver.yaml (your new source of truth)
# =========================================================
@dataclass(frozen=True)
class StakeSolverParams:
    raise_mult: List[float]
    allin_gate_spr: float

def _load_stake_solver_params(solver_yaml: Dict[str, Any], stake_key: str) -> StakeSolverParams:
    st = solver_yaml.get(stake_key) or {}
    raise_mult = st.get("raise_mult") or []
    if not isinstance(raise_mult, list) or not raise_mult:
        raise ValueError(f"{stake_key}.raise_mult missing/invalid in solver.yaml")
    raise_mult_f = [float(x) for x in raise_mult]
    gate = float(st.get("allin_gate_spr", 0.0) or 0.0)
    if gate <= 0:
        raise ValueError(f"{stake_key}.allin_gate_spr missing/invalid in solver.yaml")
    return StakeSolverParams(raise_mult=raise_mult_f, allin_gate_spr=gate)


# =========================================================
# Build solver command file for a job
# =========================================================
def _allow_allin(*, pot_bb: float, effective_stack_bb: float, gate_spr: float) -> bool:
    if pot_bb <= 0 or effective_stack_bb <= 0:
        return False
    spr = effective_stack_bb / max(pot_bb, 1e-9)
    return spr <= gate_spr + 1e-9

def _build_solver_command_text_for_job(
    *,
    params: Dict[str, Any],
    stake_params: StakeSolverParams,
    dump_path: Path,
) -> str:
    pot_bb = float(params["pot_bb"])
    eff_bb = float(params["effective_stack_bb"])
    board = str(params["board"])
    street = int(params.get("street", 1))

    size_pct = int(params["size_pct"])
    size_fr = max(0.0, min(2.0, float(size_pct) / 100.0))

    bet_sizing_id = str(params.get("bet_sizing_id") or "")
    limp = is_limp_bet_sizing_id(bet_sizing_id)

    # ranges -> monker format
    range_ip = to_monker(params["range_ip"])
    range_oop = to_monker(params["range_oop"])

    enable_allin = _allow_allin(
        pot_bb=pot_bb,
        effective_stack_bb=eff_bb,
        gate_spr=stake_params.allin_gate_spr,
    )

    sn = _street_name(street)

    bet_sizes: BetSizesType = {
        "flop": {
            "ip": {"bet": [], "raise": list(stake_params.raise_mult), "allin": enable_allin},
            "oop": {"donk": [], "bet": [], "raise": list(stake_params.raise_mult), "allin": enable_allin},
        },
        "turn": {
            "ip": {"bet": [], "raise": list(stake_params.raise_mult), "allin": enable_allin},
            "oop": {"donk": [], "bet": [], "raise": list(stake_params.raise_mult), "allin": enable_allin},
        },
        "river": {
            "ip": {"bet": [], "raise": list(stake_params.raise_mult), "allin": enable_allin},
            "oop": {"donk": [], "bet": [], "raise": list(stake_params.raise_mult), "allin": enable_allin},
        },
    }

    # IP always gets the bet menu size for the job
    bet_sizes[sn]["ip"]["bet"] = [size_fr]

    if limp:
        # ✅ Limp contract: OOP checks first, so DO NOT provide OOP donk/bet sizes at root.
        # Leaving oop["donk"] and oop["bet"] empty enforces check-only root for OOP.
        pass
    else:
        oop_kind = _oop_root_kind(bet_sizing_id)  # donk vs bet
        if oop_kind == "donk":
            bet_sizes[sn]["oop"]["donk"] = [size_fr]
        else:
            bet_sizes[sn]["oop"]["bet"] = [size_fr]

    accuracy = float(params.get("accuracy", 0.02))
    max_iter = int(params.get("max_iter", 4000))
    allin_threshold = float(params.get("allin_threshold", 0.67))

    return build_command_text(
        pot_bb=pot_bb,
        effective_stack_bb=eff_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
        bet_sizes=bet_sizes,
        allin_threshold=allin_threshold,
        thread_num=1,
        accuracy=accuracy,
        max_iteration=max_iter,
        print_interval=20,
        use_isomorphism=1,
        dump_path=str(dump_path),
    )


def _run_solver(*, solver_bin: str, cmd_file: Path) -> None:
    cmd = [solver_bin, "-i", str(cmd_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "solver failed\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc : {res.returncode}\n"
            f"--- stdout ---\n{res.stdout}\n"
            f"--- stderr ---\n{res.stderr}\n"
        )


# =========================================================
# SQS message handler
# =========================================================
def _require(params: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in params or params[k] in (None, "")]
    if missing:
        raise ValueError(f"job params missing required keys: {missing}")

def _s3_dir_of_key(s3_key: str) -> str:
    # solver/outputs/.../size=33p/output_result.json.gz -> solver/outputs/.../size=33p
    return str(s3_key).rsplit("/", 1)[0]

def _upload_json(s3, bucket: str, key: str, obj: dict) -> None:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

def _upload_text(s3, bucket: str, key: str, text: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain")

def handle_job(
    *,
    body: str,
    s3,
    bucket: str,
    solver_bin: str,
    stake_params: StakeSolverParams,
    post_upload_sleep_s: float,
    post_upload_jitter: float,
) -> bool:
    msg = json.loads(body)
    sha1 = str(msg["sha1"])
    s3_key = str(msg["s3_key"])
    params = dict(msg.get("params") or {})

    # Minimal param contract (aligned with submitter whitelist)
    _require(params,
             "street", "pot_bb", "effective_stack_bb", "board",
             "range_ip", "range_oop",
             "bet_sizing_id", "size_pct",
             "accuracy", "max_iter", "allin_threshold")

    # Idempotent: if already in S3, ack immediately
    if _exists_in_s3(s3, bucket, s3_key):
        print(f"[skip] exists: s3://{bucket}/{s3_key}")
        return True

    work_dir = Path(tempfile.mkdtemp(prefix=f"solve_{sha1[:8]}_"))
    try:
        out_json = work_dir / "output_result.json"
        cmd_file = work_dir / "commands.txt"

        cmd_text = _build_solver_command_text_for_job(
            params=params,
            stake_params=stake_params,
            dump_path=out_json,
        )
        cmd_file.write_text(cmd_text, encoding="utf-8")

        _run_solver(solver_bin=solver_bin, cmd_file=cmd_file)

        if not out_json.exists() or out_json.stat().st_size <= 0:
            raise RuntimeError("solver produced empty/missing output_result.json")

        gz_path = gzip_file(out_json)
        _upload_file_with_verify(s3, gz_path, bucket, s3_key)

        print(f"✅ uploaded: s3://{bucket}/{s3_key}")

        solve_dir = _s3_dir_of_key(s3_key)

        # 1) job.json (what was requested)
        _upload_json(s3, bucket, f"{solve_dir}/job.json", {
            "sha1": sha1,
            "s3_key": s3_key,
            "params": params,
        })

        # 2) commands.txt (what we executed)
        _upload_text(s3, bucket, f"{solve_dir}/commands.txt", cmd_text)

        # 3) resolved stake slice (what config we believe was used)
        # include only the stuff needed to verify LIMP is correct
        _upload_json(s3, bucket, f"{solve_dir}/stake_params.json", {
            "stake": getattr(stake_params, "stake_key", None),
            "raise_mult": getattr(stake_params, "raise_mult", None),
            "allin_gate_spr": getattr(stake_params, "allin_gate_spr", None),
            "bet_menus": getattr(stake_params, "bet_menus", None),
            "solver_profile": {
                "accuracy": params.get("accuracy"),
                "max_iter": params.get("max_iter"),
                "allin_threshold": params.get("allin_threshold"),
            },
        })

        # cooldown to reduce throttling
        sleep_s = max(0.0, float(post_upload_sleep_s))
        if sleep_s > 0:
            jitter = float(post_upload_jitter)
            time.sleep(sleep_s + random.random() * jitter * sleep_s)

        return True

    except Exception as e:
        print(f"❌ job failed sha1={sha1[:12]} key={s3_key} err={e}")
        return False
    finally:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


# =========================================================
# Main poll loop (simple, no legacy SQSWorker dependency)
# =========================================================
def main():
    import argparse

    ap = argparse.ArgumentParser("RangeNet postflop solver worker (SQS → solve → S3)")
    ap.add_argument("--queue-url", required=True)
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
    ap.add_argument("--bucket", default=os.getenv("AWS_BUCKET_NAME"))
    ap.add_argument("--solver-bin", default=os.getenv("SOLVER_BIN", "console_solver"))

    ap.add_argument("--solver-yaml", required=True, help="path to solver.yaml")
    ap.add_argument("--stake-key", default="Stakes.NL10", help="e.g. Stakes.NL10")

    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--wait-seconds", type=int, default=20)
    ap.add_argument("--visibility-timeout", type=int, default=120)

    ap.add_argument("--post-upload-sleep", type=float, default=float(os.getenv("POST_UPLOAD_SLEEP", "10")))
    ap.add_argument("--post-upload-jitter", type=float, default=float(os.getenv("POST_UPLOAD_JITTER", "0.25")))

    # ✅ NEW: stop conditions (default exits after a few empty polls)
    ap.add_argument(
        "--max-empty-polls",
        type=int,
        default=int(os.getenv("MAX_EMPTY_POLLS", "3")),
        help="Exit after this many consecutive empty polls (0 = never exit). Default: 3",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Do a single receive_message call and exit (useful for sanity runs)",
    )

    args = ap.parse_args()

    if not args.bucket:
        raise SystemExit("AWS_BUCKET_NAME env var or --bucket is required")

    solver_yaml = _load_yaml(args.solver_yaml)
    stake_params = _load_stake_solver_params(solver_yaml, args.stake_key)

    s3 = _s3_client(args.region)
    sqs = boto3.client("sqs", region_name=args.region)

    print(
        "✅ worker up",
        f"queue={args.queue_url}",
        f"bucket={args.bucket}",
        f"stake={args.stake_key}",
        f"threads={args.threads}",
        f"max_empty_polls={args.max_empty_polls}",
        f"once={args.once}",
        sep="\n  ",
    )

    def _process_one(m) -> Tuple[bool, str]:
        body = m["Body"]
        ok = handle_job(
            body=body,
            s3=s3,
            bucket=args.bucket,
            solver_bin=args.solver_bin,
            stake_params=stake_params,
            post_upload_sleep_s=args.post_upload_sleep,
            post_upload_jitter=args.post_upload_jitter,
        )
        return ok, m["ReceiptHandle"]

    # ✅ NEW
    empty_polls = 0

    while True:
        resp = sqs.receive_message(
            QueueUrl=args.queue_url,
            MaxNumberOfMessages=min(10, max(1, args.threads)),
            WaitTimeSeconds=args.wait_seconds,
            VisibilityTimeout=args.visibility_timeout,
        )
        msgs = resp.get("Messages") or []

        if not msgs:
            empty_polls += 1

            if args.once:
                print("ℹ️ queue empty (once=True) -> exit")
                return

            if args.max_empty_polls and empty_polls >= args.max_empty_polls:
                print(f"ℹ️ queue empty for {empty_polls} polls -> exit")
                return

            continue

        empty_polls = 0

        if args.threads <= 1 or len(msgs) == 1:
            for m in msgs:
                ok, rh = _process_one(m)
                if ok:
                    sqs.delete_message(QueueUrl=args.queue_url, ReceiptHandle=rh)
            continue

        # threaded batch
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(_process_one, m) for m in msgs]
            for fut in as_completed(futs):
                ok, rh = fut.result()
                if ok:
                    sqs.delete_message(QueueUrl=args.queue_url, ReceiptHandle=rh)


if __name__ == "__main__":
    main()
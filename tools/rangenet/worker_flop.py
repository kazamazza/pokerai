from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import json
import os
import time
import shutil
import random
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from infra.utils.gzip_file import gzip_file
from workers.base import SQSWorker

from ml.range.solvers.command_text import build_command_text
from ml.etl.utils.monker_range_converter import to_monker


# =========================
# ENV / CONSTANTS
# =========================

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
AWS_BUCKET = os.getenv("AWS_BUCKET_NAME")
SOLVER_BIN = os.getenv("SOLVER_BIN", "console_solver")

POST_UPLOAD_SLEEP = float(os.getenv("POST_UPLOAD_SLEEP", "8"))
POST_UPLOAD_JITTER = float(os.getenv("POST_UPLOAD_JITTER", "0.25"))

if not AWS_BUCKET:
    raise RuntimeError("AWS_BUCKET_NAME env var is required")


# =========================
# S3 CLIENT
# =========================

_S3CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
    tcp_keepalive=True,
)

def s3_client():
    return boto3.client("s3", region_name=AWS_REGION, config=_S3CFG)


def s3_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NotFound"):
            return False
        raise


def upload_verified(s3, local: Path, key: str) -> None:
    size = local.stat().st_size
    s3.upload_file(
        str(local),
        AWS_BUCKET,
        key,
        ExtraArgs={
            "ContentType": "application/json",
            "ContentEncoding": "gzip",
        },
    )
    head = s3.head_object(Bucket=AWS_BUCKET, Key=key)
    if int(head.get("ContentLength", -1)) != size:
        raise RuntimeError("S3 size mismatch after upload")


# =========================
# VALIDATION
# =========================

REQUIRED_PARAMS = {
    "pot_bb",
    "effective_stack_bb",
    "board",
    "range_ip",
    "range_oop",
    "size_pct",
    "accuracy",
    "max_iter",
    "allin_threshold",
}

def validate_message(msg: Dict[str, Any]) -> None:
    if "sha1" not in msg:
        raise ValueError("missing sha1")
    if "s3_key" not in msg:
        raise ValueError("missing s3_key")
    if "params" not in msg:
        raise ValueError("missing params")

    params = msg["params"]
    missing = REQUIRED_PARAMS - params.keys()
    if missing:
        raise ValueError(f"missing params: {sorted(missing)}")


# =========================
# SOLVER EXECUTION
# =========================

def run_solver(cmd_file: Path) -> None:
    proc = subprocess.run(
        [SOLVER_BIN, "-i", str(cmd_file)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "solver failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


# =========================
# MESSAGE HANDLER
# =========================

def handle_message(body: str) -> bool:
    msg = json.loads(body)
    validate_message(msg)

    sha1 = msg["sha1"]
    s3_key = msg["s3_key"]
    p = msg["params"]

    s3 = s3_client()

    # --- idempotency ---
    if s3_exists(s3, s3_key):
        print(f"[skip] already solved: {s3_key}")
        return True

    workdir = Path(tempfile.mkdtemp(prefix=f"pf_{sha1[:8]}_"))
    try:
        out_json = workdir / "result.json"
        cmd_txt = workdir / "command.txt"

        size_pct = int(p["size_pct"])
        size_fr = size_pct / 100.0

        bet_sizes = {
            "flop": {
                "ip":  {"bet": [size_fr], "raise": [], "allin": True},
                "oop": {"bet": [size_fr], "raise": [], "allin": True},
            },
            "turn":  {"ip": {"bet": []}, "oop": {"bet": []}},
            "river": {"ip": {"bet": []}, "oop": {"bet": []}},
        }

        cmd = build_command_text(
            pot_bb=float(p["pot_bb"]),
            effective_stack_bb=float(p["effective_stack_bb"]),
            board=str(p["board"]),
            range_ip=to_monker(p["range_ip"]),
            range_oop=to_monker(p["range_oop"]),
            bet_sizes=bet_sizes,
            accuracy=float(p["accuracy"]),
            max_iteration=int(p["max_iter"]),
            allin_threshold=float(p["allin_threshold"]),
            dump_path=str(out_json),
        )

        cmd_txt.write_text(cmd, encoding="utf-8")

        run_solver(cmd_txt)

        if not out_json.exists() or out_json.stat().st_size == 0:
            raise RuntimeError("solver produced no output")

        gz = gzip_file(out_json)
        upload_verified(s3, gz, s3_key)

        sleep_s = POST_UPLOAD_SLEEP * (1 + random.random() * POST_UPLOAD_JITTER)
        time.sleep(sleep_s)

        print(f"✅ solved {sha1} → s3://{AWS_BUCKET}/{s3_key}")
        return True

    except Exception as e:
        print(f"❌ job {sha1} failed: {e}")
        return False

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# =========================
# ENTRYPOINT
# =========================

def main():
    worker = SQSWorker(
        handler=handle_message,
        queue_url=os.getenv("POSTFLOP_QUEUE_URL"),
        dlq_url=os.getenv("POSTFLOP_DLQ_URL"),
        region=AWS_REGION,
        batch_size=1,
        max_threads=1,
    )
    worker.run()


if __name__ == "__main__":
    main()
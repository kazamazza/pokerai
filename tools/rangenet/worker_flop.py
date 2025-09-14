from __future__ import annotations
import os, json, tempfile, shutil, subprocess, uuid
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import boto3
import numpy as np
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.monker_range_converter import to_monker
from ml.etl.utils.range_lookup import monker_string_to_vec169

from dotenv import load_dotenv

load_dotenv()

from workers.base import SQSWorker
from ml.range.solvers.command_text import build_command_text

# --------- Config knobs (env or fallback) ----------
SOLVER_BIN = os.getenv("SOLVER_BIN", "external/worker/console_solver")
LOCAL_CACHE_DIR = Path(os.getenv("SOLVER_LOCAL_CACHE", "data/solver_cache")).resolve()
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")  # required at runtime
REGION = os.getenv("AWS_REGION", "eu-central-1")

BET_MENUS: Dict[str, Dict[str, Dict[str, Dict[str, Union[list[int], bool]]]]] = {
    "std": {
        "flop": {
            "oop": {"donk": [25], "bet": [33, 50, 75], "raise": [66, 100, 150], "allin": True},
            "ip":  {"bet": [25, 33, 50, 75], "raise": [66, 100, 150], "allin": True},
        },
        "turn": {
            "oop": {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
            "ip":  {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
        },
        "river": {
            "oop": {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
            "ip":  {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
        },
    }
}

def _format_bet_sizes(
    menu_id: Optional[str]
) -> Optional[Dict[str, Dict[str, Dict[str, Union[list[int], bool]]]]]:
    # Default to "std", and if an unknown id is passed, fall back to "std"
    key = (menu_id or "std")
    return BET_MENUS.get(key, BET_MENUS.get("std"))


def _exists_in_s3(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return False
        # treat all other errors as transient (do not ack message)
        raise


def _upload_file(s3, local_path: Path, bucket: str, key: str) -> None:
    local_path = Path(local_path)
    s3.upload_file(str(local_path), bucket, key)


def _nnz_stats_from_payload(payload) -> tuple[int, float]:
    # try to view it as a 169-vector (0..1)
    try:
        if isinstance(payload, str):
            obj = json.loads(payload) if payload.strip().startswith(("{","[")) else None
        else:
            obj = payload
        if isinstance(obj, list):
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 2 and arr.shape == (13,13):
                arr = arr.reshape(169)
            if arr.ndim == 1 and arr.size == 169:
                arr = np.clip(arr, 0.0, 1.0)
                return int((arr > 0).sum()), float(arr.sum())
    except Exception:
        pass
    return -1, -1.0


def _nnz_and_sum_from_monker(s: str) -> tuple[int, float]:
    try:
        v = monker_string_to_vec169(s)
        nnz = sum(1 for x in v if x > 1e-9)
        return nnz, float(sum(v))
    except Exception:
        return -1, -1.0

def _build_solver_cmd_text(
    params: Dict[str, Any],
    dump_path: Path,
    job_id: str = "noid"
) -> Path:
    pot_bb  = float(params["pot_bb"])
    eff_bb  = float(params["effective_stack_bb"])
    board   = str(params["board"])

    # --- pre-conversion stats ---
    pre_ip_nnz, pre_ip_sum = _nnz_stats_from_payload(params["range_ip"])
    pre_oop_nnz, pre_oop_sum = _nnz_stats_from_payload(params["range_oop"])
    print(f"[range-stats:pre] IP nnz={pre_ip_nnz} sum={pre_ip_sum:.2f} | OOP nnz={pre_oop_nnz} sum={pre_oop_sum:.2f}")

    # Convert to Monker string
    range_ip  = to_monker(params["range_ip"])
    range_oop = to_monker(params["range_oop"])

    # --- post-conversion stats ---
    post_ip_nnz, post_ip_sum = _nnz_and_sum_from_monker(range_ip)
    post_oop_nnz, post_oop_sum = _nnz_and_sum_from_monker(range_oop)
    print(f"[range-stats:post] IP nnz={post_ip_nnz} sum={post_ip_sum:.2f} | OOP nnz={post_oop_nnz} sum={post_oop_sum:.2f}")

    # Guards
    if post_ip_nnz != -1 and post_ip_nnz < 10:
        raise RuntimeError(f"IP range too sparse post-conversion (nnz={post_ip_nnz}) – source looks wrong")
    if post_oop_nnz != -1 and post_oop_nnz < 10:
        raise RuntimeError(f"OOP range too sparse post-conversion (nnz={post_oop_nnz}) – source looks wrong")

    bet_menu_id = str(params.get("bet_sizing_id", "std"))
    bet_sizes = _format_bet_sizes(bet_menu_id)

    accuracy = float(params.get("accuracy", 0.5))
    max_iter = int(params.get("max_iter", params.get("max_iterations", 200)))
    a_th = float(params.get("allin_threshold", 0.67))

    txt = build_command_text(
        pot_bb=pot_bb,
        effective_stack_bb=eff_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
        bet_sizes=bet_sizes,
        allin_threshold=a_th,
        thread_num=1,
        accuracy=accuracy,
        max_iteration=max_iter,
        print_interval=10,
        use_isomorphism=0,
        dump_path=str(dump_path),
    )

    # Write the command file in the job’s work_dir instead of debug_cmds/
    cmd_path = Path(dump_path).parent / f"{job_id}_commands.txt"
    cmd_path.write_text(txt, encoding="utf-8")
    print(f"📝 wrote solver command → {cmd_path}")

    return cmd_path


def _run_solver(cmd_file: Path) -> None:
    """
    Run the external console_solver with the given command file.
    On failure, include the command file path and contents for debugging.
    """
    import os
    solver_bin = Path(SOLVER_BIN)
    cmd_file = Path(cmd_file)

    cmd = [str(solver_bin), "-i", str(cmd_file)]
    print(f"▶️ worker cmd: {' '.join(cmd)}")
    print(f"    cwd: {os.getcwd()}")
    try:
        size = cmd_file.stat().st_size
    except Exception:
        size = -1
    print(f"    cmd_file: {cmd_file.resolve()}  ({size} bytes)")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # read command file content for debugging
        try:
            content = cmd_file.read_text()
        except Exception as e:
            content = f"<unable to read command file: {e}>"

        # truncate if massive
        MAX_SHOW = 20000
        shown = content if len(content) <= MAX_SHOW else (content[:MAX_SHOW] + "\n…<truncated>…\n")

        raise RuntimeError(
            "worker failed\n"
            f"  code: {result.returncode}\n"
            f"  cmd : {' '.join(cmd)}\n"
            f"  file: {cmd_file.resolve()} ({size} bytes)\n"
            "\n--- command file contents ---\n"
            f"{shown}"
            "\n--- worker stdout ---\n"
            f"{result.stdout}"
            "\n--- worker stderr ---\n"
            f"{result.stderr}"
        )


def handle_message(body: str) -> bool:
    """
    SQSWorker handler: receives raw message body (string),
    returns True to delete from queue, False/None to leave (DLQ handled by base).
    """
    if not S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET env var is required")

    msg = json.loads(body)
    sha1   = msg["sha1"]
    s3_key = msg["s3_key"]
    params = msg["params"]

    # Idempotency: if result already exists, ack immediately.
    s3 = boto3.client("s3", region_name=REGION)
    if _exists_in_s3(s3, S3_BUCKET, s3_key):
        print(f"[skip] already in S3: s3://{S3_BUCKET}/{s3_key}")
        return True

    work_dir = Path(tempfile.mkdtemp(prefix=f"solve_{sha1[:8]}_"))
    try:
        out_json = work_dir / "result.json"

        # Build command file (function writes it and returns its path)
        cmd_path = _build_solver_cmd_text(params, out_json, job_id=sha1)

        # Run solver
        _run_solver(cmd_path)

        # Sanity: result file must exist and be non-empty
        if not out_json.exists() or out_json.stat().st_size == 0:
            raise RuntimeError("worker produced no result.json or file empty")

        # Upload to S3
        _upload_file(s3, out_json, S3_BUCKET, s3_key)
        print(f"✅ uploaded → s3://{S3_BUCKET}/{s3_key}")
        return True

    except Exception as e:
        print(f"❌ job {sha1} failed: {e}")
        return False
    finally:
        # Clean up temp files
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


def main():
    import argparse

    ap = argparse.ArgumentParser(description="RangeNet FLOP worker (SQS)")
    ap.add_argument("--queue-url", type=str, default=os.getenv("POST_FLOP_QUEUE_URL"), required=False)
    ap.add_argument("--dlq-url",   type=str, default=os.getenv("POST_FLOP_DLQ_URL"),   required=False)
    ap.add_argument("--region",    type=str, default=os.getenv("AWS_REGION", "eu-central-1"))
    ap.add_argument("--batch-size", type=int, default=1)        # single message at a time works well
    ap.add_argument("--threads",    type=int, default=1)        # SQS worker threads; set 1 for vCPU
    args = ap.parse_args()

    worker = SQSWorker(
        handler=handle_message,
        max_threads=args.threads,
        batch_size=args.batch_size,
        region=args.region,
        queue_url=args.queue_url,
        dlq_url=args.dlq_url,
    )
    worker.run()  # assumes your base class exposes run()

if __name__ == "__main__":
    main()
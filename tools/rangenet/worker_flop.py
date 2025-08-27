from __future__ import annotations
import os, json, tempfile, shutil, subprocess, uuid
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import boto3
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from workers.base import SQSWorker
from ml.range.solvers.command_text import build_command_text

# --------- Config knobs (env or fallback) ----------
SOLVER_BIN = os.getenv("SOLVER_BIN", "external/solver/console_solver")
LOCAL_CACHE_DIR = Path(os.getenv("SOLVER_LOCAL_CACHE", "data/solver_cache")).resolve()
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")  # required at runtime
REGION = os.getenv("AWS_REGION", "eu-central-1")

# Optional: small bet menu map; extend as needed
BET_MENUS: Dict[str, Dict[str, Dict[str, Dict[str, list[int]]]]] = {
    "std": {
        "flop": {
            "oop": {
                "bet":   [50],   # one standard c-bet size (50% pot)
                "raise": [66],   # one standard raise size (~2/3 pot)
                "donk":  [],     # OOP donk disabled for now
                "allin": True    # always allow all-in
            },
            "ip": {
                "bet":   [50],   # one standard bet size in position
                "raise": [66],   # one standard raise size (~2/3 pot)
                "donk":  [],     # IP donk is nonsensical
                "allin": True
            },
        },
    },
}


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


def _format_bet_sizes(menu_id: Optional[str]) -> Optional[Dict[str, Dict[str, Dict[str, list[int]]]]]:
    if not menu_id:
        return BET_MENUS.get("std")
    return BET_MENUS.get(menu_id, BET_MENUS.get("std"))


def _build_solver_cmd_text(params: Dict[str, Any], dump_path: Path) -> str:
    """
    Build the solver command text from the message params.
    Required params for FLOP:
      pot_bb, effective_stack_bb, board, range_ip, range_oop
    Optional:
      bet_sizing_id, accuracy, max_iter, allin_threshold
    """
    pot_bb  = float(params["pot_bb"])
    eff_bb  = float(params["effective_stack_bb"])
    board   = str(params["board"])
    range_ip  = str(params["range_ip"])
    range_oop = str(params["range_oop"])

    bet_menu_id = str(params.get("bet_sizing_id", "std"))
    bet_sizes = _format_bet_sizes(bet_menu_id)

    accuracy = float(params.get("accuracy", 0.5))
    max_iter = int(params.get("max_iter", 200))
    a_th     = float(params.get("allin_threshold", 0.67))

    txt = build_command_text(
        pot_bb=pot_bb,
        effective_stack_bb=eff_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
        bet_sizes=bet_sizes,
        allin_threshold=a_th,
        thread_num=1,                 # single-thread for vCPU
        accuracy=accuracy,
        max_iteration=max_iter,
        print_interval=10,
        use_isomorphism=1,
        dump_path=str(dump_path),
    )
    return txt


def _run_solver(cmd_file: Path) -> None:
    """
    Run the external console_solver with the given command file.
    On failure, include the command file path and contents for debugging.
    """
    import os
    solver_bin = Path(SOLVER_BIN)
    cmd_file = Path(cmd_file)

    cmd = [str(solver_bin), "-i", str(cmd_file)]
    print(f"▶️ solver cmd: {' '.join(cmd)}")
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
            "solver failed\n"
            f"  code: {result.returncode}\n"
            f"  cmd : {' '.join(cmd)}\n"
            f"  file: {cmd_file.resolve()} ({size} bytes)\n"
            "\n--- command file contents ---\n"
            f"{shown}"
            "\n--- solver stdout ---\n"
            f"{result.stdout}"
            "\n--- solver stderr ---\n"
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
    sha1 = msg["sha1"]
    s3_key = msg["s3_key"]
    params = msg["params"]

    # Idempotency: if result already exists, ack immediately.
    s3 = boto3.client("s3", region_name=REGION)
    if _exists_in_s3(s3, S3_BUCKET, s3_key):
        print(f"[skip] already in S3: s3://{S3_BUCKET}/{s3_key}")
        return True

    # Work dir
    work_dir = Path(tempfile.mkdtemp(prefix=f"solve_{sha1[:8]}_"))
    try:
        out_json = work_dir / "result.json"
        cmd_txt  = work_dir / "commands.txt"

        # Build command text
        cmd_text = _build_solver_cmd_text(params, out_json)
        cmd_txt.write_text(cmd_text)

        # Run solver
        _run_solver(cmd_txt)

        # Sanity: result file present & nonempty
        if not out_json.exists() or out_json.stat().st_size == 0:
            raise RuntimeError("solver produced no result.json or file empty")

        # Upload
        _upload_file(s3, out_json, S3_BUCKET, s3_key)
        print(f"✅ uploaded → s3://{S3_BUCKET}/{s3_key}")
        return True

    except Exception as e:
        # Log full error; returning False lets the SQSWorker move to DLQ per its policy.
        print(f"❌ job {sha1} failed: {e}")
        return False
    finally:
        # Clean up temp files to save disk
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

    print(args)

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
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.utils.gzip_file import gzip_file
from ml.etl.utils.monker_range_converter import to_monker
from ml.etl.utils.range_lookup import monker_string_to_vec169
from ml.config.bet_menus import build_contextual_bet_sizes
from dotenv import load_dotenv

load_dotenv()

from workers.base import SQSWorker
from ml.range.solvers.command_text import build_command_text

import os, json, time, gzip, shutil, tempfile, subprocess, random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# === ENV / constants ===
REGION    = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")
SOLVER_BIN = os.getenv("SOLVER_BIN", "console_solver")

# Post-upload pacing (seconds); jitter added automatically
POST_UPLOAD_SLEEP = float(os.getenv("POST_UPLOAD_SLEEP", "10"))   # base seconds
POST_UPLOAD_JITTER = float(os.getenv("POST_UPLOAD_JITTER", "0.25"))

# -------- S3 client with adaptive retries --------
_S3CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=60,
    tcp_keepalive=True,
)

def _s3_client(region: str):
    return boto3.client("s3", region_name=region, config=_S3CFG)


# -------- context helpers (unchanged logic) --------
def _ctx_from_menu(menu_id: str) -> str:
    m = (menu_id or "").lower()
    if m.startswith("limped_single"): return "LIMPED_SINGLE"
    if m.startswith("limped_multi"):  return "LIMPED_MULTI"
    if m.startswith("3bet_hu"):       return "VS_3BET"
    if m.startswith("4bet_hu"):       return "VS_4BET"
    return "SRP"

_MIN_NNZ_WARN = {
    "SRP": 20,
    "VS_3BET": 12,
    "VS_4BET": 2,
    "LIMPED_SINGLE": 8,
    "LIMPED_MULTI": 8,
}

def _sparse_decision(ip_nnz: int, oop_nnz: int, ctx: str) -> tuple[bool, str]:
    if ip_nnz == 0 or oop_nnz == 0:
        return True, f"empty range (ip={ip_nnz}, oop={oop_nnz})"
    floor = _MIN_NNZ_WARN.get(ctx, 8)
    if ip_nnz < floor or oop_nnz < floor:
        return False, f"ranges sparse for {ctx} (ip={ip_nnz}, oop={oop_nnz} < {floor}); continuing"
    return False, ""

def _exists_in_s3(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code"))
        http = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NotFound") or http == 404:
            return False
        raise

def _nnz_stats_from_payload(payload) -> tuple[int, float]:
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

# -------- robust upload with verification --------
TRANSIENT_HTTP = {429, 500, 502, 503, 504}
TRANSIENT_CODES = {
    "SlowDown", "Throttling", "ThrottlingException",
    "RequestTimeout", "RequestTimeoutException",
    "InternalError", "ServiceUnavailable"
}

def _upload_file_with_verify(
    s3, local_path: Path, bucket: str, key: str,
    *, content_type: Optional[str] = None,
    content_encoding: Optional[str] = None,
    max_tries: int = 7
) -> None:
    local_path = Path(local_path)
    if not local_path.exists():
        raise RuntimeError(f"Local file missing: {local_path}")
    size = local_path.stat().st_size

    extra = {}
    if content_type:     extra["ContentType"] = content_type
    if content_encoding: extra["ContentEncoding"] = content_encoding

    backoff = 0.5
    for attempt in range(1, max_tries + 1):
        try:
            s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra if extra else None)
            head = s3.head_object(Bucket=bucket, Key=key)
            if int(head.get("ContentLength", -1)) == size or size <= 0:
                return
            print(f"[warn] S3 size mismatch (local={size}, remote={head.get('ContentLength')}); retrying…")
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code"))
            http = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            transient = (http in TRANSIENT_HTTP) or (code in TRANSIENT_CODES)
            if not transient or attempt == max_tries:
                raise
        except Exception:
            if attempt == max_tries:
                raise
        time.sleep(backoff + random.random()*0.3)
        backoff = min(10.0, backoff * 2)
    raise RuntimeError("S3 upload failed after retries/verify")

# -------- solver IO --------
def _build_solver_cmd_text(params: Dict[str, Any], dump_path: Path, job_id: str = "noid") -> tuple[str, Path]:
    pot_bb  = float(params["pot_bb"])
    eff_bb  = float(params["effective_stack_bb"])
    board   = str(params["board"])

    pre_ip_nnz,  pre_ip_sum  = _nnz_stats_from_payload(params["range_ip"])
    pre_oop_nnz, pre_oop_sum = _nnz_stats_from_payload(params["range_oop"])
    print(f"[range-stats:pre] IP nnz={pre_ip_nnz} sum={pre_ip_sum:.2f} | OOP nnz={pre_oop_nnz} sum={pre_oop_sum:.2f}")

    range_ip  = to_monker(params["range_ip"])
    range_oop = to_monker(params["range_oop"])

    post_ip_nnz,  post_ip_sum  = _nnz_and_sum_from_monker(range_ip)
    post_oop_nnz, post_oop_sum = _nnz_and_sum_from_monker(range_oop)
    print(f"[range-stats:post] IP nnz={post_ip_nnz} sum={post_ip_sum:.2f} | OOP nnz={post_oop_nnz} sum={post_oop_sum:.2f}")

    menu_id = str(params.get("bet_sizing_id", "") or "")
    ctx = _ctx_from_menu(menu_id)
    if post_ip_nnz != -1 and post_oop_nnz != -1:
        fail, msg = _sparse_decision(post_ip_nnz, post_oop_nnz, ctx)
        if fail: raise RuntimeError(msg + f" – ctx={ctx}")
        elif msg: print("[warn]", msg)

    bet_sizes = build_contextual_bet_sizes(menu_id)

    accuracy = float(params.get("accuracy", 0.5))
    max_iter = int(params.get("max_iter", params.get("max_iterations", 200)))
    a_th     = float(params.get("allin_threshold", 0.67))

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
        print_interval=20,
        use_isomorphism=1,
        dump_path=str(dump_path),
    )

    dbg_dir = Path("debug_cmds") / job_id
    dbg_dir.mkdir(parents=True, exist_ok=True)
    cmd_path = dbg_dir / "commands.txt"
    cmd_path.write_text(txt, encoding="utf-8")

    def _clip(s): return s if len(s) <= 400 else s[:400] + "..."
    (dbg_dir / "ranges.txt").write_text(
        "RAW range_ip:\n"
        f"{str(params['range_ip'])[:400]}\n\n"
        "RAW range_oop:\n"
        f"{str(params['range_oop'])[:400]}\n\n"
        "MONKER range_ip:\n"
        f"{_clip(range_ip)}\n\n"
        "MONKER range_oop:\n"
        f"{_clip(range_oop)}\n",
        encoding="utf-8"
    )

    print(f"📝 dumped solver command → {cmd_path}")
    return txt, cmd_path

def _run_solver(cmd_file: Path) -> None:
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
        try:
            content = cmd_file.read_text()
        except Exception as e:
            content = f"<unable to read command file: {e}>"
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

# -------- SQS handler (bulletproof ack only after verified upload) --------
def handle_message(body: str) -> bool:
    if not S3_BUCKET:
        raise RuntimeError("AWS_BUCKET_NAME env var is required")
    print("incoming message:", body)
    msg = json.loads(body)
    sha1 = msg["sha1"]
    s3_key = msg["s3_key"]
    params = msg["params"]

    s3 = _s3_client(REGION)

    # Fast idempotency: already uploaded?
    if _exists_in_s3(s3, S3_BUCKET, s3_key):
        print(f"[skip] already in S3: s3://{S3_BUCKET}/{s3_key}")
        return True

    work_dir = Path(tempfile.mkdtemp(prefix=f"solve_{sha1[:8]}_"))
    try:
        out_json = work_dir / "result.json"

        # Build command + run
        cmd_text, cmd_txt = _build_solver_cmd_text(params, out_json, job_id=sha1)
        cmd_txt.write_text(cmd_text)
        _run_solver(cmd_txt)

        # Sanity output
        if not out_json.exists() or out_json.stat().st_size == 0:
            raise RuntimeError("solver produced no result.json or file empty")

        # GZip and verified upload with retries
        gz_path = gzip_file(out_json)
        _upload_file_with_verify(
            s3, gz_path, S3_BUCKET, s3_key,
            content_type="application/json",
            content_encoding="gzip",
        )
        print(f"✅ uploaded → s3://{S3_BUCKET}/{s3_key}")

        # Pace before acking / taking another ticket
        sleep_s = max(0.0, POST_UPLOAD_SLEEP) + random.random() * POST_UPLOAD_JITTER * POST_UPLOAD_SLEEP
        if sleep_s > 0:
            print(f"⏳ cooling down {sleep_s:.2f}s to ease S3 throttling…")
            time.sleep(sleep_s)

        return True

    except Exception as e:
        print(f"❌ job {sha1} failed: {e}")
        # Don’t ack; let SQS redrive/DLQ handle it
        return False
    finally:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

def main():
    import argparse
    ap = argparse.ArgumentParser(description="RangeNet FLOP worker (SQS, hardened uploads)")
    ap.add_argument("--queue-url", type=str, default=os.getenv("POST_FLOP_QUEUE_URL"))
    ap.add_argument("--dlq-url",   type=str, default=os.getenv("POST_FLOP_DLQ_URL"))
    ap.add_argument("--region",    type=str, default=REGION)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--threads",    type=int, default=1)
    ap.add_argument("--post-upload-sleep", type=float, default=10,
                    help="Seconds to sleep after a verified upload (base, jitter applied)")
    args = ap.parse_args()

    # allow runtime override of sleep
    global POST_UPLOAD_SLEEP
    POST_UPLOAD_SLEEP = float(args.post_upload_sleep)

    worker = SQSWorker(
        handler=handle_message,
        max_threads=args.threads,
        batch_size=args.batch_size,
        region=args.region,
        queue_url=args.queue_url,
        dlq_url=args.dlq_url,
    )
    worker.run()

if __name__ == "__main__":
    main()
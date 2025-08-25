# ml/etl/rangenet/postflop/worker_handler.py
from __future__ import annotations

import json
import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path

from infra.storage.s3_client import S3Client
from ml.range.solvers.command_text import build_command_text

# Resolve repo root → solver binary
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOLVER_BIN = (PROJECT_ROOT / "external/solver/console_solver").resolve()

def _s3_key_exists(s3: S3Client, s3_key: str) -> bool:
    """
    Cheap existence check. Uses list prefix on the parent folder and exact match.
    Replace with head_object if you prefer.
    """
    parent = "/".join(s3_key.split("/")[:-1])
    for k in s3.list_files(prefix=parent + "/" if parent else ""):
        if k == s3_key:
            return True
    return False


def handle_solve_message(body: str) -> bool:
    """
    Adapter for SQSWorker(handler=...). Returns True on success (uploaded exists),
    False on failure (DLQ will handle).
    Expects message body like:
      {
        "sha1": "...",
        "s3_key": "solver/outputs/v1/<sha1>/output_result.json.gz",
        "params": {
          "street": 1,
          "pot_bb": 20,
          "effective_stack_bb": 200,
          "board": "QsJh2h",
          "range_ip": "...",
          "range_oop": "...",
          "positions": "OOPvIP",
          "bet_sizing_id": "std",
          "accuracy": 0.5,
          "max_iter": 200,
          "allin_threshold": 0.67,
          "node_key": "root",
          "solver_version": "v1"
        }
      }
    """
    try:
        msg = json.loads(body)
        s3_key = msg["s3_key"]
        params = msg["params"]

        # S3 client (bucket from env or constructor)
        s3 = S3Client()

        # 1) Short-circuit if already uploaded
        if _s3_key_exists(s3, s3_key):
            print(f"[worker] cache hit on S3: s3://{s3.bucket}/{s3_key}")
            return True

        # 2) Create a temp run dir
        work_root = Path("data/solver_work")
        work_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"solve_{msg['sha1'][:8]}_", dir=work_root))
        try:
            out_json = run_dir / "output_result.json"
            in_txt   = run_dir / "input.txt"

            # 3) Build the solver command file (force absolute dump path!)
            cmd_text = build_command_text(
                pot_bb=float(params["pot_bb"]),
                effective_stack_bb=float(params["effective_stack_bb"]),
                board=str(params["board"]) if params.get("board") else "",
                range_ip=str(params["range_ip"]),
                range_oop=str(params["range_oop"]),
                bet_sizes=str(params["bet_sizing_id"]),
                accuracy=float(params["accuracy"]),
                max_iteration=int(params["max_iter"]),
                allin_threshold=float(params["allin_threshold"]),
                positions=str(params["positions"]),
                street=int(params["street"]),
                dump_path=str(out_json.resolve()),  # 🔑 force absolute path for dump_result
            )
            in_txt.write_text(cmd_text, encoding="utf-8")

            # 4) Run the console solver in that folder
            print(f"[worker] running solver in {run_dir} …")
            proc = subprocess.run(
                [str(SOLVER_BIN), "-i", "input.txt"],
                cwd=str(run_dir),
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                print(f"[worker] solver stderr:\n{proc.stderr}")
                print(f"[worker] solver stdout:\n{proc.stdout}")
                return False

            # 5) Verify output
            if not out_json.exists():
                print("[worker] ❌ solver finished but output_result.json missing")
                print(f"[worker] stdout:\n{proc.stdout}")
                print(f"[worker] stderr:\n{proc.stderr}")
                return False

            # 6) Gzip and upload
            gz_path = out_json.with_suffix(".json.gz")
            with out_json.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            s3.upload_file(gz_path, s3_key)
            # sanity: check it exists now
            return _s3_key_exists(s3, s3_key)

        finally:
            # always clean local run dir
            shutil.rmtree(run_dir, ignore_errors=True)

    except Exception as e:
        print(f"[worker] ❌ exception in handler: {e}")
        return False
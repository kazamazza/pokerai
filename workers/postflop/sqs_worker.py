# postflop/sqs_worker.py
import os
import io
import gzip
import json
import time
import traceback
from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# repo wiring
from pathlib import Path as _P
ROOT_DIR = _P(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT_DIR))

from workers.base import SQSWorker
from utils.keys import build_postflop_key, build_preflop_s3_key

# If you already have a generator that wraps your solver, import that instead:
# from postflop.generate.generate_cluster_strategy import generate_cluster_strategy

load_dotenv()

REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
BUCKET = os.getenv("AWS_BUCKET_NAME") or "pokeraistore"

s3 = boto3.client("s3", region_name=REGION)

# ---- solver adapter ---------------------------------------------------------
# Replace this with your actual solver call (Piosolver/TexasSolver/etc).
def solve_postflop_strategy(
    *,
    cluster_id: int,
    stack_bb: int,
    ip_position: str,
    oop_position: str,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str,
    preflop_json: dict,
) -> dict:
    """
    Return a JSON-serializable strategy dict.
    Plug your solver invocation here.
    """
    # Example if you already have a wrapper:
    # model = generate_cluster_strategy(
    #   cluster_id=cluster_id, stack_bb=stack_bb,
    #   ip_position=ip_position, oop_position=oop_position,
    #   villain_profile=villain_profile, exploit_setting=exploit_setting,
    #   multiway_context=multiway_context, population_type=population_type,
    #   action_context=action_context
    # )
    # try: return model.model_dump()
    # except Exception: return model.dict()
    # For now, stub so file shape is predictable:
    return {
        "meta": {
            "cluster_id": cluster_id,
            "stack_bb": stack_bb,
            "ip_position": ip_position,
            "oop_position": oop_position,
            "villain_profile": villain_profile,
            "exploit_setting": exploit_setting,
            "multiway_context": multiway_context,
            "population_type": population_type,
            "action_context": action_context,
        },
        "strategy": {
            # fill with your solver outputs
        },
    }

# ---- helpers ----------------------------------------------------------------

def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code", "")
        return False if code in ("404", "NoSuchKey", "NotFound") else False

def fetch_preflop_json_from_s3(cfg: dict) -> dict:
    """
    Read the preflop artifact directly from S3 into memory and return parsed JSON.
    No local file is created.
    """
    key = build_preflop_s3_key(cfg)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
    except ClientError as e:
        raise RuntimeError(f"preflop missing: s3://{BUCKET}/{key} ({e})")

    by = obj["Body"].read()  # bytes (gzipped JSON)
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        return json.loads(gz.read().decode("utf-8"))

def load_preflop_json(local_gz: Path) -> dict:
    with open(local_gz, "rb") as f:
        by = f.read()
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        return json.loads(gz.read().decode("utf-8"))

# ---- handler ----------------------------------------------------------------

def handle_postflop_task(message_body: str) -> bool:
    """
    Message schema expected (same as your producer):
      {
        "cluster_id": int,
        "ip_position": str, "oop_position": str, "stack_bb": int,
        "villain_profile": str, "exploit_setting": str,
        "multiway_context": str, "population_type": str, "action_context": str
      }
    Returns True only when uploaded & verified.
    """
    try:
        cfg = json.loads(message_body)
        out_key = build_postflop_key(cfg)

        # Idempotency: skip if the output already exists
        if s3_exists(out_key):
            print(f"🟢 exists: s3://{BUCKET}/{out_key}")
            return True

        # Ensure preflop dependency is local + load it for the solver
        preflop_json = fetch_preflop_json_from_s3(cfg)


        # Run solver / generator
        strategy_json = solve_postflop_strategy(preflop_json=preflop_json, **cfg)

        # Write local file first (use same filename your key implies)
        out_dir = Path("postflop/strategy_templates") / \
            cfg["villain_profile"] / cfg["exploit_setting"] / \
            cfg["multiway_context"] / cfg["population_type"] / cfg["action_context"]
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{cfg['ip_position']}_vs_{cfg['oop_position']}_{int(cfg['stack_bb'])}bb_cluster_{int(cfg['cluster_id'])}.json"
        local_out = out_dir / out_name
        with open(local_out, "w") as f:
            json.dump(strategy_json, f, indent=2)

        # Upload + verify
        s3.upload_file(str(local_out), BUCKET, out_key)
        for _ in range(5):
            try:
                head = s3.head_object(Bucket=BUCKET, Key=out_key)
                if head.get("ContentLength", 0) > 0:
                    print(f"✅ uploaded: s3://{BUCKET}/{out_key}")
                    return True
            except Exception as e:
                print(f"⚠️ head retry: {e}")
            time.sleep(0.5)

        print(f"❌ verify failed: s3://{BUCKET}/{out_key}")
        return False

    except Exception as e:
        print(f"❌ postflop task error: {e}")
        traceback.print_exc()
        return False

# ---- entry ------------------------------------------------------------------

if __name__ == "__main__":
    # Keep conservative to avoid visibility/locking headaches
    worker = SQSWorker(
        handler=handle_postflop_task,   # returns bool -> base worker will only delete on True
        max_threads=1,
        batch_size=1,
        # region/urls come from env:
        #   AWS_REGION / AWS_SQS_QUEUE_URL / AWS_SQS_DLQ_URL
    )
    worker.run()
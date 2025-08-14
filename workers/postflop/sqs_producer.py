# postflop/producer.py
import os, json, itertools, time
from pathlib import Path
from typing import Dict, Iterator
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from utils.keys import build_postflop_key

ROOT_DIR = Path(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT_DIR))

from features.types import STACK_BUCKETS
from preflop.matchups import MATCHUPS

# --- Env / AWS ---
load_dotenv()
REGION   = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
BUCKET   = os.getenv("AWS_BUCKET_NAME") or "pokeraistore"
QUEUE_URL = os.getenv("POSTFLOP_QUEUE_URL")
SKIP_IF_EXISTS = os.getenv("SKIP_IF_EXISTS") in ("1","true","True","YES","yes")

s3  = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)

# --- Cluster map (determine valid cluster ids) ---
with open(ROOT_DIR / "data/flop/flop_cluster_map.json", "r") as f:
    FLOP_CLUSTER_MAP = json.load(f)

VALID_CLUSTERS = sorted(set(FLOP_CLUSTER_MAP.values()))

# Fixed knobs (same as your local generator)
VILLAIN_PROFILE  = "GTO"
EXPLOIT_SETTING  = "GTO"
MULTIWAY_CONTEXT = "HU"
POPULATION_TYPE  = "REGULAR"
ACTION_CONTEXT   = "OPEN"


def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code", "")
        return False if code in ("404","NoSuchKey","NotFound") else False

def build_all_tasks() -> Iterator[Dict]:
    for cluster_id in VALID_CLUSTERS:
        for ip, oop in MATCHUPS:
            for stack in STACK_BUCKETS:
                yield {
                    "cluster_id":       int(cluster_id),
                    "ip_position":      ip,
                    "oop_position":     oop,
                    "stack_bb":         int(stack),
                    "villain_profile":  VILLAIN_PROFILE,
                    "exploit_setting":  EXPLOIT_SETTING,
                    "multiway_context": MULTIWAY_CONTEXT,
                    "population_type":  POPULATION_TYPE,
                    "action_context":   ACTION_CONTEXT,
                }

def enqueue_all(dry_run: bool = False) -> None:
    if not QUEUE_URL:
        raise SystemExit("❌ POSTFLOP_QUEUE_URL not set.")

    total = 0
    sent = 0
    batch = []
    started = time.time()

    for task in build_all_tasks():
        total += 1

        # Optional idempotency skip: if the OUTPUT already exists, skip enqueue
        if SKIP_IF_EXISTS and s3_exists(build_postflop_key(task["cluster_id"], task["ip_position"], task["oop_position"], task["stack_bb"])):
            if total % 1000 == 0:
                print(f"⏭️  Skipped {total} (exists)")
            continue

        batch.append({
            "Id": str(total % 10 or 10),
            "MessageBody": json.dumps(task),
        })

        if len(batch) == 10:
            if not dry_run:
                resp = sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=batch)
                failed = resp.get("Failed", [])
                sent += len(batch) - len(failed)
            else:
                sent += len(batch)
            batch = []

        if total % 1000 == 0:
            rate = total / max(1.0, (time.time()-started))
            print(f"🟢 Prepared {total} tasks (~{rate:.1f}/s)")

    if batch:
        if not dry_run:
            resp = sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=batch)
            failed = resp.get("Failed", [])
            sent += len(batch) - len(failed)
        else:
            sent += len(batch)

    elapsed = time.time() - started
    rate = sent / max(1.0, elapsed)
    print(f"✅ Enqueued {sent}/{total} in {elapsed:.1f}s (~{rate:.1f}/s)")

if __name__ == "__main__":
    dry = os.getenv("DRY_RUN") in ("1","true","True","YES","yes")
    enqueue_all(dry_run=dry)
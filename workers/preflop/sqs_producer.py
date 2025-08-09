import os
import json
import sys
import time
from pathlib import Path
from typing import List, Iterator, Dict
import boto3
import itertools
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from features.types import VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS, \
    STACK_BUCKETS
from preflop.matchups import MATCHUPS
from utils.keys import build_preflop_s3_key

# Load AWS credentials from .env
load_dotenv()
sqs = boto3.client(
    "sqs",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

REGION    = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")
sqs       = boto3.client("sqs", region_name=REGION)

VALID_POS = {"BTN","SB","BB","UTG","MP","CO"}

def _validate_cfg(cfg: Dict) -> None:
    if cfg["ip_position"] not in VALID_POS or cfg["oop_position"] not in VALID_POS:
        raise ValueError(f"Invalid positions: {cfg['ip_position']} vs {cfg['oop_position']}")
    if int(cfg["stack_bb"]) not in STACK_BUCKETS:
        raise ValueError(f"Invalid stack_bb: {cfg['stack_bb']}")

def build_all_configs() -> Iterator[Dict]:
    """
    Yields the exact payload the worker expects.
    """
    for profile, exploit, multiway, pop, action in itertools.product(
        VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS
    ):
        for ip, oop in MATCHUPS:
            for stack in STACK_BUCKETS:
                cfg = {
                    "ip_position": ip,
                    "oop_position": oop,
                    "stack_bb": int(stack),
                    "villain_profile": profile,
                    "exploit_setting": exploit,
                    "multiway_context": multiway,
                    "population_type": pop,
                    "action_context": action
                }
                # sanity: build the key now so producer/worker/validator stay aligned
                _ = build_preflop_s3_key(cfg)
                yield cfg

def enqueue_all_configs(dry_run: bool = False) -> None:
    if not QUEUE_URL:
        raise SystemExit("❌ PRE_FLOP_QUEUE_URL env var is not set.")

    total_expected = (
        len(VILLAIN_PROFILES)
        * len(EXPLOIT_SETTINGS)
        * len(MULTIWAY_CONTEXTS)
        * len(POPULATION_TYPES)
        * len(ACTION_CONTEXTS)
        * len(MATCHUPS)
        * len(STACK_BUCKETS)
    )
    print(f"📦 Expected total to enqueue: {total_expected}")

    # Batch up to 10 messages per API call
    batch: List[Dict] = []
    sent = 0
    sample_key_printed = False
    batch_id = 0

    def flush():
        nonlocal batch, sent, batch_id
        if not batch:
            return
        if dry_run:
            sent += len(batch)
            batch = []
            return
        try:
            resp = sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=batch)
            failed = resp.get("Failed", [])
            if failed:
                print(f"❌ Batch {batch_id} had {len(failed)} failures: {failed}")
            sent += len(batch) - len(failed)
        except ClientError as e:
            print(f"❌ send_message_batch error (batch {batch_id}): {e}")
        finally:
            batch = []

    entry_ctr = 0
    started = time.time()

    for i, cfg in enumerate(build_all_configs(), 1):
        try:
            _validate_cfg(cfg)
        except Exception as e:
            print(f"⚠️ Skipping invalid cfg at #{i}: {e}")
            continue

        # Print one sample resolved S3 key once, for sanity
        if not sample_key_printed:
            print("🔎 Sample S3 key for this payload:",
                  build_preflop_s3_key(cfg))
            sample_key_printed = True

        entry_ctr += 1
        batch_id = (i // 10)
        batch.append({
            "Id": f"id-{entry_ctr % 10 or 10}",  # unique per-batch
            "MessageBody": json.dumps(cfg),
        })

        if len(batch) == 10:
            flush()

        if i % 1000 == 0:
            rate = i / max(1.0, (time.time() - started))
            print(f"🟢 Enqueued {i} (rate ~{rate:.1f}/s)")

    flush()

    elapsed = time.time() - started
    rate = sent / max(1.0, elapsed)
    print(f"✅ Done. Total enqueued: {sent} in {elapsed:.1f}s (~{rate:.1f}/s)")
    if sent != total_expected:
        print(f"🟨 Note: expected {total_expected}, actually enqueued {sent} "
              f"(skipped invalid or dry-run?).")

if __name__ == "__main__":
    dry = os.getenv("DRY_RUN") in ("1","true","True","YES","yes")
    enqueue_all_configs(dry_run=dry)
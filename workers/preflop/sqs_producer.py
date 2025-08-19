import os
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Counter
import boto3
import itertools
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from preflop.preflop_jobs_for_tuple import preflop_jobs_for_tuple
from utils.poker import POSITION_ORDER
from utils.ec2 import is_ec2_instance, shutdown_instance

load_dotenv()
REGION    = os.getenv("AWS_REGION", "eu-central-1")
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")
if not QUEUE_URL:
    raise RuntimeError("PRE_FLOP_QUEUE_URL is not set")

sqs = boto3.client(
    "sqs",
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

from ml.types import (
     EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES,
    STACK_BUCKETS, POSITIONS, ACTION_CONTEXTS, RAKE_TIERS, ANTE_BB, OPEN_SIZE_POLICIES
)

POSITION_ORDER = POSITIONS
INDEX = {p: i for i, p in enumerate(POSITION_ORDER)}

def iter_producer_messages() -> Iterator[dict]:
    for expl, mw, pop in itertools.product(
        EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES
    ):
        for stack in STACK_BUCKETS:
            for rake in RAKE_TIERS:
                for ante in ANTE_BB:
                    for open_policy in OPEN_SIZE_POLICIES:
                        for ip in POSITION_ORDER:
                            for oop in POSITION_ORDER:
                                if ip == oop:
                                    continue
                                for ctx in ACTION_CONTEXTS:
                                    for msg in preflop_jobs_for_tuple(
                                         ctx,
                                        ip,
                                        oop,
                                        stack,
                                        expl,
                                        mw,
                                        pop,
                                        rake_tier=rake,
                                        ante_bb=ante,
                                        open_size_policy=open_policy,
                                    ):
                                        yield msg


def _full_key(cfg):
    return (
        cfg["action_context"],
        cfg["ip_position"],
        cfg["oop_position"],
        cfg["stack_bb"],
        cfg["exploit_setting"],
        cfg["multiway_context"],
        cfg["population_type"],
        cfg.get("rake_tier"),
        cfg.get("ante_bb"),
        cfg.get("open_size_policy"),
    )

def dry_run_producer():
    totals_by_ctx = Counter()
    unique_by_ctx = defaultdict(set)
    combos = defaultdict(list)

    breakdown_by_stack = defaultdict(lambda: Counter())
    breakdown_by_rake  = defaultdict(lambda: Counter())

    for cfg in iter_producer_messages():
        k = _full_key(cfg)
        ctx = cfg["action_context"]
        totals_by_ctx[ctx] += 1
        unique_by_ctx[ctx].add(k)
        combos[k].append(cfg)

        # extra breakdowns
        breakdown_by_stack[ctx][cfg["stack_bb"]] += 1
        breakdown_by_rake[ctx][cfg["rake_tier"]] += 1

    # Report totals
    print("=== Coverage Report ===")
    for ctx in sorted(totals_by_ctx.keys()):
        total = totals_by_ctx[ctx]
        uniq  = len(unique_by_ctx[ctx])
        print(f"\n{ctx}: {total} jobs | unique={uniq}")

        # Show breakdowns
        print("  By Stack:")
        for s, c in sorted(breakdown_by_stack[ctx].items()):
            print(f"    {s}bb: {c}")

        print("  By Rake Tier:")
        for r, c in breakdown_by_rake[ctx].items():
            print(f"    {r}: {c}")

    print(f"\nTOTAL jobs across all contexts: {sum(totals_by_ctx.values())}")

def enqueue_all_configs():
    seen = set()
    total = 0
    for cfg in iter_producer_messages():
        k = (
            cfg["action_context"], cfg["ip_position"], cfg["oop_position"], cfg["stack_bb"],
            cfg["villain_profile"], cfg["exploit_setting"], cfg["multiway_context"], cfg["population_type"]
        )
        if k in seen:
            continue
        seen.add(k)

        sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(cfg))
        total += 1
        if total % 1000 == 0:
            print(f"🟢 Enqueued {total} tasks...")
    print(f"✅ Done. Total tasks enqueued: {total}")

if __name__ == "__main__":
    dry_run_producer()
    if is_ec2_instance():
        shutdown_instance()
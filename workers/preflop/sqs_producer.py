import os
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, List

import boto3
import itertools
from dotenv import load_dotenv

from utils.poker import POSITION_ORDER

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

load_dotenv()

sqs = boto3.client(
    "sqs",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")

from features.types import VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS, \
    STACK_BUCKETS, POSITIONS
from utils.ec2 import is_ec2_instance, shutdown_instance

# Load AWS credentials from .env
load_dotenv()

INDEX = {p:i for i,p in enumerate(POSITION_ORDER)}

def is_valid_open_pair(opener: str, defender: str) -> bool:
    return INDEX[opener] < INDEX[defender]

# Contexts you actually want to produce

# ---- Minimal pairs per context (what files the cluster extractor will expect)
def preflop_jobs_for_tuple(context: str, ip: str, oop: str, stack: int,
                           prof: str, expl: str, mw: str, pop: str) -> List[Dict]:
    """
    Return the *exact* messages to enqueue so that the cluster stage
    can later build ranges for this (context, ip, oop, stack, axes).
    """
    msgs: List[Dict] = []

    if context == "OPEN":
        # Need: OPEN(ip,oop) and VS_OPEN(oop,ip). Only if opener is earlier than defender.
        if is_valid_open_pair(ip, oop):
            msgs.append({
                "ip_position": ip, "oop_position": oop, "stack_bb": stack,
                "villain_profile": prof, "exploit_setting": expl,
                "multiway_context": mw, "population_type": pop,
                "action_context": "OPEN"
            })
            msgs.append({
                "ip_position": oop, "oop_position": ip, "stack_bb": stack,
                "villain_profile": prof, "exploit_setting": expl,
                "multiway_context": mw, "population_type": pop,
                "action_context": "VS_OPEN"
            })
        # else: skip invalid OPEN pairing

    elif context == "VS_OPEN":
        # If you want direct VS_OPEN clusters too (same SRP tuple), you still
        # need the same pair as OPEN. Reuse the same two files.
        if is_valid_open_pair(ip, oop):
            msgs.extend(preflop_jobs_for_tuple("OPEN", ip, oop, stack, prof, expl, mw, pop))

    elif context == "VS_3BET":
        # Need: VS_OPEN(oop,ip) for the 3-bettor range, and VS_3BET(ip,oop) for opener’s call vs 3bet.
        msgs.append({
            "ip_position": oop, "oop_position": ip, "stack_bb": stack,
            "villain_profile": prof, "exploit_setting": expl,
            "multiway_context": mw, "population_type": pop,
            "action_context": "VS_OPEN"     # defender perspective vs opener
        })
        msgs.append({
            "ip_position": ip, "oop_position": oop, "stack_bb": stack,
            "villain_profile": prof, "exploit_setting": expl,
            "multiway_context": mw, "population_type": pop,
            "action_context": "VS_3BET"     # opener response vs 3bet
        })

    elif context == "VS_4BET":
        # Need: VS_3BET(ip,oop) for opener’s 4bet range, and VS_4BET(oop,ip) for defender’s call vs 4bet.
        msgs.append({
            "ip_position": ip, "oop_position": oop, "stack_bb": stack,
            "villain_profile": prof, "exploit_setting": expl,
            "multiway_context": mw, "population_type": pop,
            "action_context": "VS_3BET"     # opener’s 4bet lives here
        })
        msgs.append({
            "ip_position": oop, "oop_position": ip, "stack_bb": stack,
            "villain_profile": prof, "exploit_setting": expl,
            "multiway_context": mw, "population_type": pop,
            "action_context": "VS_4BET"     # defender’s call vs 4bet
        })

    elif context == "VS_LIMP":
        # Keep symmetric and cheap: require both POVs of VS_LIMP.
        msgs.append({
            "ip_position": ip, "oop_position": oop, "stack_bb": stack,
            "villain_profile": prof, "exploit_setting": expl,
            "multiway_context": mw, "population_type": pop,
            "action_context": "VS_LIMP"
        })
        msgs.append({
            "ip_position": oop, "oop_position": ip, "stack_bb": stack,
            "villain_profile": prof, "exploit_setting": expl,
            "multiway_context": mw, "population_type": pop,
            "action_context": "VS_LIMP"
        })

    return msgs

# ---- The actual producer
def iter_producer_messages() -> Iterator[Dict]:
    for prof, expl, mw, pop in itertools.product(
        VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES
    ):
        for stack in STACK_BUCKETS:
            for ip in POSITIONS:
                for oop in POSITIONS:
                    if ip == oop:
                        continue
                    for ctx in ACTION_CONTEXTS:
                        for msg in preflop_jobs_for_tuple(ctx, ip, oop, stack, prof, expl, mw, pop):
                            yield msg

def enqueue_all_configs():
    total = 0
    for cfg in iter_producer_messages():
        sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(cfg))
        total += 1
        if total % 1000 == 0:
            print(f"🟢 Enqueued {total} tasks...")
    print(f"✅ Done. Total tasks enqueued: {total}")

if __name__ == "__main__":
    enqueue_all_configs()
    if is_ec2_instance():
        shutdown_instance()
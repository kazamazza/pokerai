import os
import json
import sys
from pathlib import Path
import boto3
import itertools
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.expected_counts import update_expected_count
from features.types import VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS, \
    STACK_BUCKETS
from preflop.matchups import MATCHUPS
from utils.ec2 import is_ec2_instance, shutdown_instance

# Load AWS credentials from .env
load_dotenv()
sqs = boto3.client(
    "sqs",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# Your SQS queue URL
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")

def build_all_configs():
    for profile, exploit, multiway, pop, action in itertools.product(
        VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS
    ):
        for ip, oop in MATCHUPS:
            for stack in STACK_BUCKETS:
                yield {
                    "ip_position": ip,
                    "oop_position": oop,
                    "stack_bb": stack,
                    "villain_profile": profile,
                    "exploit_setting": exploit,
                    "multiway_context": multiway,
                    "population_type": pop,
                    "action_context": action
                }


def enqueue_all_configs():
    total = 0
    for config in build_all_configs():
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(config)
        )
        total += 1
        if total % 100 == 0:
            print(f"🟢 Enqueued {total} tasks...")
    update_expected_count("preflop", total)
    print(f"✅ Done. Total tasks enqueued: {total}")


if __name__ == "__main__":
    enqueue_all_configs()
    if is_ec2_instance():
        shutdown_instance()
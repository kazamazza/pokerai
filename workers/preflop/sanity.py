import itertools
import json
import os
from typing import Iterator

import boto3
from dotenv import load_dotenv
from features.types import VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS, \
    STACK_BUCKETS
from preflop.matchups import MATCHUPS
# Load AWS credentials from .env if needed
load_dotenv()

s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
BUCKET = os.getenv("AWS_BUCKET_NAME")

def build_all_keys() -> Iterator[str]:
    for profile, exploit, multiway, pop, action in itertools.product(
        VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS
    ):
        for ip, oop in MATCHUPS:
            for stack in STACK_BUCKETS:
                key = (
                    f"preflop/ranges/"
                    f"profile={profile}/exploit={exploit}/multiway={multiway}/"
                    f"pop={pop}/action={action}/"
                    f"{oop}_vs_{ip}_{stack}bb.json.gz"
                )
                yield key

def check_s3_keys_exist():
    print("🔍 Starting sanity check...")
    missing = []
    total = 0
    for key in build_all_keys():
        total += 1
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"❌ MISSING: {key}")
                missing.append(key)
            else:
                print(f"⚠️ Error checking {key}: {e}")

    print(f"\n✅ Sanity check complete.")
    print(f"Total expected: {total}")
    print(f"Missing files : {len(missing)}")

    if missing:
        with open("missing_preflop_keys.json", "w") as f:
            json.dump(missing, f, indent=2)
        print("📄 Missing keys written to missing_preflop_keys.json")

if __name__ == "__main__":
    check_s3_keys_exist()

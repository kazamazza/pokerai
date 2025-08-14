import json
import os
import re
from typing import Dict

import boto3
import dotenv

dotenv.load_dotenv()

REGION = os.getenv("AWS_REGION", "eu-central-1")
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")
sqs = boto3.client("sqs", region_name=REGION)

# Use the SAME regex that matches your canonical key shape (IP_vs_OOP etc.)
KEY_RX = re.compile(
    r"^preflop/ranges/"
    r"profile=(?P<profile>[^/]+)/exploit=(?P<exploit>[^/]+)/multiway=(?P<multiway>[^/]+)/"
    r"pop=(?P<pop>[^/]+)/action=(?P<action>[^/]+)/"
    r"(?P<ip>BTN|SB|BB|UTG|MP|CO)_vs_(?P<oop>BTN|SB|BB|UTG|MP|CO)_(?P<stack>\d+)bb\.json\.gz$"
)

def key_to_cfg(key: str) -> Dict:
    m = KEY_RX.match(key)
    if not m:
        raise ValueError(f"Bad key shape: {key}")
    d = m.groupdict()
    return {
        "ip_position": d["ip"],
        "oop_position": d["oop"],
        "stack_bb": int(d["stack"]),
        "villain_profile": d["profile"],
        "exploit_setting": d["exploit"],
        "multiway_context": d["multiway"],
        "population_type": d["pop"],
        "action_context": d["action"],
    }

def main(path="missing_keys.txt"):
    if not QUEUE_URL:
        raise SystemExit("PRE_FLOP_QUEUE_URL is not set")

    lines = [ln.strip() for ln in open(path) if ln.strip()]
    print(f"Re‑enqueueing {len(lines)} missing…")

    batch, sent = [], 0
    def flush():
        nonlocal batch, sent
        if not batch: return
        resp = sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=batch)
        failed = resp.get("Failed", [])
        if failed:
            print(f"Batch had {len(failed)} failures: {failed[:2]}…")
        sent += len(batch) - len(failed)
        batch.clear()

    idx = 0
    for k in lines:
        try:
            cfg = key_to_cfg(k)
        except Exception as e:
            print(f"Skip unparsable key {k}: {e}")
            continue

        idx += 1
        batch.append({"Id": f"id-{idx%10 or 10}", "MessageBody": json.dumps(cfg)})
        if len(batch) == 10:
            flush()

        if idx % 1000 == 0:
            print(f"Queued {idx} / {len(lines)}")

    flush()
    print(f"✅ Re‑enqueued ~{sent} tasks")

if __name__ == "__main__":
    main()
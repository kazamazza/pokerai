import os, json, re, itertools, time, threading
from typing import Iterator, Dict, List, Tuple
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- project imports ---
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT_DIR))
from ml.types import (
    VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS, STACK_BUCKETS
)
from preflop.matchups import MATCHUPS

load_dotenv()

REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
BUCKET = os.getenv("AWS_BUCKET_NAME") or "pokeraistore"
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")  # main preflop queue

# tunables via env
WORKERS = int(os.getenv("CHECK_WORKERS", "32"))          # concurrent S3 head requests
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "500")) # print progress every N keys
LIMIT = int(os.getenv("LIMIT", "0")) or None             # cap for testing
DRY_RUN = os.getenv("DRY_RUN", "").lower() in {"1","true","yes"}

s3  = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)

KEY_RX = re.compile(
    r"^preflop/ranges/"
    r"profile=(?P<profile>[^/]+)/exploit=(?P<exploit>[^/]+)/multiway=(?P<multiway>[^/]+)/"
    r"pop=(?P<pop>[^/]+)/action=(?P<action>[^/]+)/"
    r"(?P<oop>BTN|SB|BB|UTG|MP|CO)_vs_(?P<ip>BTN|SB|BB|UTG|MP|CO)_(?P<stack>\d+)bb\.json\.gz$"
)

def build_all_keys() -> Iterator[str]:
    for profile, exploit, multiway, pop, action in itertools.product(
        VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS
    ):
        for ip, oop in MATCHUPS:
            for stack in STACK_BUCKETS:
                yield (
                    "preflop/ranges/"
                    f"profile={profile}/exploit={exploit}/multiway={multiway}/"
                    f"pop={pop}/action={action}/"
                    f"{ip}_vs_{oop}_{stack}bb.json.gz"
                )

def key_to_config(key: str) -> Dict:
    m = KEY_RX.match(key)
    if not m:
        raise ValueError(f"Key does not match expected pattern: {key}")
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

def s3_key_exists(bucket: str, key: str) -> Tuple[bool, str | None]:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head.get("ContentLength", 0) <= 0:
            return False, "zero-size"
        return True, None
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False, "not-found"
        return False, f"head_object:{code}"

def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def enqueue_missing(missing_keys: List[str]) -> None:
    if not missing_keys:
        print("🎉 No missing items to re-enqueue.")
        return
    if not QUEUE_URL:
        print("❌ PRE_FLOP_QUEUE_URL not set; cannot re-enqueue.")
        return

    print(f"📨 Re-enqueueing {len(missing_keys)} tasks to {QUEUE_URL} …")
    batch: List[Dict] = []
    sent = 0; batch_num = 0; eid = 0

    def flush():
        nonlocal batch, sent, batch_num
        if not batch: return
        batch_num += 1
        resp = sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=batch)
        failed = resp.get("Failed", [])
        if failed:
            print(f"❌ Batch {batch_num}: {len(failed)} failed: {failed}")
        sent += len(batch) - len(failed)
        batch = []

    for key in missing_keys:
        try:
            cfg = key_to_config(key)
        except Exception as e:
            print(f"⚠️  Skip unparsable key: {key} ({e})")
            continue
        eid += 1
        batch.append({"Id": f"id-{eid % 10 or 10}", "MessageBody": json.dumps(cfg)})
        if len(batch) == 10:
            flush()
    flush()
    print(f"✅ Re-enqueued ~{sent} tasks.")

def check_and_reenqueue():
    # materialize expected keys
    keys = list(build_all_keys())
    if LIMIT: keys = keys[:LIMIT]
    total = len(keys)
    print(f"🔍 Scanning S3 for missing preflop artifacts …")
    print(f"Bucket: s3://{BUCKET}/preflop/ranges/ | Expected: {total} | Workers: {WORKERS}")

    start = time.time()
    done = 0
    done_lock = threading.Lock()
    missing: List[str] = []
    reasons: Dict[str, int] = {}

    def _record(k: str, ok: bool, why: str | None):
        nonlocal done
        with done_lock:
            done += 1
            if not ok:
                missing.append(k)
                if why: reasons[why] = reasons.get(why, 0) + 1
            if (done % PROGRESS_EVERY == 0) or (done == total):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = _fmt_eta((total - done) / rate) if rate > 0 else "∞"
                pct = (done / total * 100.0) if total else 100.0
                print(f"⏱️  {done}/{total} ({pct:5.1f}%) "
                      f"| ok {done - len(missing)} | miss {len(missing)} "
                      f"| {rate:5.1f} obj/s | ETA {eta}")

    # parallel HEADs
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(s3_key_exists, BUCKET, k): k for k in keys}
        for fut in as_completed(futures):
            k = futures[fut]
            ok, why = fut.result()
            _record(k, ok, why)

    dur = time.time() - start
    rate = total / dur if dur > 0 else 0.0
    print("\n✅ Scan complete.")
    print(f"⏲️  Took {dur:.1f}s | Avg {rate:.1f} obj/s")
    if reasons:
        print("Missing reasons:", reasons)
    print(f"Missing files: {len(missing)} / {total}")

    # Save list for audit
    with open("missing_preflop_keys.json", "w") as f:
        json.dump(missing, f, indent=2)
    print("📝 Wrote missing_preflop_keys.json")

    if not DRY_RUN and missing:
        enqueue_missing(missing)
    elif DRY_RUN and missing:
        print("🟨 Dry-run: not re-enqueueing.")

if __name__ == "__main__":
    check_and_reenqueue()
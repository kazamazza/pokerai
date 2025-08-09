# validate_preflop_bucket.py
import os, re, json, gzip, io, sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError  # you already use this above
import boto3
from botocore.exceptions import ClientError

REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
PREFIX = "preflop/ranges/"

s3 = boto3.client("s3", region_name=REGION)

# OOP_vs_IP_100bb.json.gz  AND path …/profile=…/exploit=…/multiway=…/pop=…/action=…/
KEY_RX = re.compile(
    r"^preflop/ranges/"
    r"profile=(?P<profile>[^/]+)/exploit=(?P<exploit>[^/]+)/multiway=(?P<multiway>[^/]+)/"
    r"pop=(?P<pop>[^/]+)/action=(?P<action>[^/]+)/"
    r"(?P<oop>BTN|SB|BB|UTG|MP|CO)_vs_(?P<ip>BTN|SB|BB|UTG|MP|CO)_(?P<stack>\d+)bb\.json\.gz$"
)

HAND_RX = re.compile(r"^(?:[2-9TJQKA]{2}(?:[so])?)$")  # e.g. 22, AKs, AKo, KQo, etc.

def list_keys(prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def load_json_gz(key: str):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    by = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        return json.loads(gz.read().decode("utf-8"))

def validate_actions(actions: dict) -> tuple[bool, str]:
    if not isinstance(actions, dict) or not actions:
        return False, "actions not dict/nonempty"
    total = 0
    for k, v in actions.items():
        if not isinstance(v, list):
            return False, f"actions[{k}] not list"
        for hand in v:
            if not isinstance(hand, str) or not HAND_RX.match(hand):
                return False, f"bad hand '{hand}' in {k}"
        total += len(v)
    if total == 0:
        return False, "no hands across actions"
    return True, ""

def _fmt_eta(seconds: float) -> str:
    if seconds < 0: seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def _check_one(key: str):
    """Return (status, bad_sample_or_None). Status matches your stats keys."""
    m = KEY_RX.match(key)
    if not m:
        return "bad_key_shape", {"key": key, "reason": "key regex mismatch"}

    # quick size check
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        if head.get("ContentLength", 0) <= 0:
            return "zero_size", {"key": key, "reason": "zero size"}
    except ClientError as e:
        return "json_error", {"key": key, "reason": f"head_object: {e}"}

    # load & validate JSON
    try:
        doc = load_json_gz(key)
    except Exception as e:
        return "json_error", {"key": key, "reason": f"json/gzip error: {e}"}

    if not isinstance(doc, dict) or "meta" not in doc or "actions" not in doc:
        return "schema_error", {"key": key, "reason": "missing meta/actions"}

    meta = doc["meta"]
    need = [
        "ip_position","oop_position","stack_bb",
        "villain_profile","exploit_setting",
        "multiway_context","population_type","action_context"
    ]
    if not all(k in meta for k in need):
        return "schema_error", {"key": key, "reason": "meta fields missing"}

    gd = m.groupdict()
    try:
        stack_ok = int(gd["stack"]) == int(meta["stack_bb"])
        names_ok = (
            gd["ip"] == meta["ip_position"]
            and gd["oop"] == meta["oop_position"]
            and gd["profile"] == meta["villain_profile"]
            and gd["exploit"] == meta["exploit_setting"]
            and gd["multiway"] == meta["multiway_context"]
            and gd["pop"] == meta["population_type"]
            and gd["action"] == meta["action_context"]
        )
    except Exception:
        stack_ok = False
        names_ok = False

    if not (stack_ok and names_ok):
        return "meta_mismatch", {"key": key, "reason": "meta vs filename mismatch", "meta": meta}

    ok, why = validate_actions(doc["actions"])
    if not ok:
        return "action_error", {"key": key, "reason": f"actions invalid: {why}"}

    return "ok", None

def main(limit: int | None = None, max_workers: int = 16, progress_every: int = 500):
    stats = {
        "scanned": 0,
        "ok": 0,
        "bad_key_shape": 0,
        "json_error": 0,
        "schema_error": 0,
        "meta_mismatch": 0,
        "action_error": 0,
        "zero_size": 0,
    }
    bad_samples = []

    # materialize list (lets us show %/ETA)
    keys = list(list_keys(PREFIX))
    if limit:
        keys = keys[:limit]
    total = len(keys)
    print(f"Bucket: s3://{BUCKET}/{PREFIX}")
    print(f"📦 Keys to validate: {total}")

    start = time.time()
    done = 0
    done_lock = threading.Lock()
    bad_lock = threading.Lock()

    def _record(result):
        nonlocal done
        status, bad = result
        with done_lock:
            stats["scanned"] += 1
            stats[status] += 1
            done = stats["scanned"]
            if done % progress_every == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = total - done
                eta = remaining / rate if rate > 0 else float("inf")
                pct = (done / total * 100) if total else 100.0
                print(f"⏱️  {done}/{total} ({pct:5.1f}%) "
                      f"| ok {stats['ok']} | bad {done - stats['ok']} "
                      f"| {rate:5.1f} obj/s | ETA {_fmt_eta(eta)}")
        if bad:
            with bad_lock:
                if len(bad_samples) < 200:
                    bad_samples.append(bad)

    # parallel validation
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_check_one, k) for k in keys]
        for f in as_completed(futures):
            _record(f.result())

    duration = time.time() - start
    avg = total / duration if duration > 0 else 0.0
    print("\n✅ Validation complete.")
    print(f"⏲️  Took: {duration:.1f}s | Avg: {avg:.1f} obj/s")
    for k, v in stats.items():
        print(f"{k:>16}: {v}")

    if bad_samples:
        with open("preflop_validation_report.json", "w") as f:
            json.dump({"stats": stats, "bad": bad_samples}, f, indent=2)
        print("Wrote preflop_validation_report.json (first 200 bad samples).")
        sys.exit(2)
    else:
        print("🎉 All checked objects look consistent.")
        sys.exit(0)

if __name__ == "__main__":
    lim = int(os.getenv("LIMIT", "0")) or None
    workers = int(os.getenv("VALIDATE_WORKERS", "16"))
    main(limit=lim, max_workers=workers, progress_every=500)
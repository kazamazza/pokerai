import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import boto3
from boto3 import s3

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client("s3", region_name=REGION)

from preflop.generate_ranges import generate_single_range
from workers.base import SQSWorker

def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def handle_preflop_task(message_body: str) -> None:
    """
    Return True only when the final S3 object is confirmed present.
    """
    try:
        config = json.loads(message_body)
        # generate_single_range should now RETURN (s3_key, temp_path)
        s3_key, temp_path = generate_single_range(config)  # <-- adjust implementation below

        # Upload (idempotent-safe: overwrite same key)
        s3.upload_file(str(temp_path), BUCKET, s3_key)

        # Verify: HEAD the key; optionally check size > 0
        for _ in range(5):
            try:
                head = s3.head_object(Bucket=BUCKET, Key=s3_key)
                size = head.get("ContentLength", 0)
                if size and size > 0:
                    print(f"✅ Verified in S3: s3://{BUCKET}/{s3_key} ({size} bytes)")
                    # cleanup local
                    Path(temp_path).unlink(missing_ok=True)
                    return True
            except Exception as e:
                print(f"⚠️ head_object retry: {e}")
            time.sleep(1.0)

        print(f"❌ Verification failed for s3://{BUCKET}/{s3_key}")
        return False

    except Exception as e:
        print(f"❌ Task failed: {e}")
        traceback.print_exc()
        return False


# preflop/sqs_worker.py
if __name__ == "__main__":
    worker = SQSWorker(
        handler=handle_preflop_task,
        max_threads=1,   # was os.cpu_count()
        batch_size=1     # smaller batch reduces timeout risk
    )
    worker.run()

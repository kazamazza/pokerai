import json
import os
import sys
import time
import traceback
from pathlib import Path
import boto3
from boto3 import s3
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client("s3", region_name=REGION)

from preflop.generate_ranges import generate_single_range
from workers.base import SQSWorker

def handle_preflop_task(message_body: str) -> bool:
    try:
        cfg = json.loads(message_body)

        s3_key, temp_path = generate_single_range(cfg)  # make this return (key, tmp_path)
        # Or: s3_key = build_preflop_key(cfg) if you build it here instead

        # Idempotency guard: if it exists, consider done
        try:
            head = s3.head_object(Bucket=BUCKET, Key=s3_key)
            if head.get("ContentLength", 0) > 0:
                print(f"🟢 Already exists: s3://{BUCKET}/{s3_key} — skipping upload")
                Path(temp_path).unlink(missing_ok=True)
                return True
        except ClientError as e:
            # 404 -> proceed to upload
            pass

        # Upload
        s3.upload_file(str(temp_path), BUCKET, s3_key)

        # Verify
        for _ in range(5):
            try:
                head = s3.head_object(Bucket=BUCKET, Key=s3_key)
                if head.get("ContentLength", 0) > 0:
                    print(f"✅ Verified: s3://{BUCKET}/{s3_key}")
                    Path(temp_path).unlink(missing_ok=True)
                    return True
            except Exception as e:
                print(f"⚠️ head_object retry: {e}")
            time.sleep(0.5)

        print(f"❌ Verification failed: s3://{BUCKET}/{s3_key}")
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

import sys
import os
import time
import traceback
from pathlib import Path
import boto3
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

load_dotenv()
from utils.files import compress_json_gzip
from generate_equity_simulations import generate_simulation
from workers.base import SQSWorker

REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client("s3", region_name=REGION)

def handle_equity_task(_: str) -> bool:
    """
    Generate one equity sample, upload gzipped JSON to S3, and verify it exists.
    Return True on success so SQSWorker will delete the message.
    """
    try:
        features, label = generate_simulation()

        # Pydantic v1/v2 compatible dict extraction
        feat_to_dict = getattr(features, "model_dump", getattr(features, "dict"))
        lab_to_dict  = getattr(label, "model_dump", getattr(label, "dict"))
        data = {
            "features": feat_to_dict(),
            "label": lab_to_dict(),
        }

        key = f"equity/simulations/{features.hash()}.json.gz"

        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=compress_json_gzip(data),
            ContentEncoding="gzip",
            ContentType="application/json",
        )
        print(f"✅ Uploaded: {key}")

        # Verify the object is present and non-empty
        for _ in range(3):
            try:
                head = s3.head_object(Bucket=BUCKET, Key=key)
                size = head.get("ContentLength", 0)
                if size and size > 0:
                    print(f"✅ Verified in S3: s3://{BUCKET}/{key} ({size} bytes)")
                    return True
            except Exception as e:
                print(f"⚠️ head_object retry for {key}: {e}")
            time.sleep(0.5)

        print(f"❌ Verification failed for s3://{BUCKET}/{key}")
        return False

    except Exception as e:
        print(f"❌ Failed to process equity task: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    worker = SQSWorker(
        handler=handle_equity_task,
        max_threads=1,
        batch_size=1,
        region="eu-central-1",
        queue_url="https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-queue",
        dlq_url="https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-dlq"
    )
    worker.run()

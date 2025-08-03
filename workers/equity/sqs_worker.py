import sys
import os
import traceback
from pathlib import Path
import boto3

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.files import compress_json_gzip
from generate_equity_simulations import generate_simulation
from workers.base import SQSWorker

BUCKET = os.environ["AWS_BUCKET_NAME"]

s3 = boto3.client("s3")

def handle_equity_task(_: str):
    features, label = generate_simulation()
    data = {
        "features": features.model_dump(),
        "label": label.model_dump()
    }
    key = f"equity/simulations/{features.hash()}.json.gz"
    try:
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=compress_json_gzip(data),
            ContentEncoding="gzip",
            ContentType="application/json"
        )
        print(f"✅ Uploaded: {key}")
    except Exception as e:
        print(f"❌ Failed to upload {key}: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    worker = SQSWorker(
        handler=handle_equity_task,
        max_threads=10,
        batch_size=10
    )
    worker.run()

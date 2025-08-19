import os
import sys
from pathlib import Path
import boto3

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client("s3", region_name=REGION)

from preflop.generate_ranges import generate_single_range
from workers.base import SQSWorker

def handle_preflop_task(message_body: str) -> bool:
    pass


# preflop/sqs_worker.py
if __name__ == "__main__":
    worker = SQSWorker(
        handler=handle_preflop_task,
        max_threads=1,   # was os.cpu_count()
        batch_size=1     # smaller batch reduces timeout risk
    )
    worker.run()

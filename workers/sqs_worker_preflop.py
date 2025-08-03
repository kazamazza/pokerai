import os
import json
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from preflop.generate_ranges import generate_single_range
from workers.base import SQSWorker

REGION = os.getenv("AWS_REGION")
QUEUE_URL = os.getenv("AWS_SQS_QUEUE_URL")
DLQ_URL = os.getenv("AWS_SQS_DLQ_URL")  # Optional fallback


def handle_preflop_task(message_body):
    try:
        config = json.loads(message_body)
        generate_single_range(config)
    except Exception as e:
        print(f"❌ Task failed: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    worker = SQSWorker(
        handler=handle_preflop_task,
        max_threads=5,
        batch_size=5
    )
    worker.run()

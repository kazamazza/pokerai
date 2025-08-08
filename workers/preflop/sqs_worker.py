import json
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from preflop.generate_ranges import generate_single_range
from workers.base import SQSWorker
from utils.threads import detect_threads

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
        max_threads=detect_threads(),  # auto
        batch_size=10
    )
    worker.run()

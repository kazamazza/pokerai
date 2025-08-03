import os
import sys
import time
import traceback
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

class SQSWorker:
    def __init__(self, region_env="AWS_REGION", queue_env="AWS_SQS_QUEUE_URL", dlq_env="AWS_SQS_DLQ_URL"):
        self.region = os.environ[region_env]
        self.queue_url = os.environ[queue_env]
        self.dlq_url = os.getenv(dlq_env)
        self.sqs = boto3.client("sqs", region_name=self.region)

    def handle_message(self, message_body: str):
        """Override this in your subclass."""
        raise NotImplementedError

    def move_to_dlq(self, message_body: str):
        if not self.dlq_url:
            print("⚠️ DLQ URL not set. Cannot forward failed message.")
            return
        try:
            self.sqs.send_message(
                QueueUrl=self.dlq_url,
                MessageBody=message_body
            )
            print("➡️ Moved failed task to DLQ")
        except ClientError as e:
            print(f"❌ Failed to move message to DLQ: {e}")

    def _process_single(self, msg):
        receipt_handle = msg["ReceiptHandle"]
        body = msg["Body"]
        try:
            print("🔄 Processing task...")
            self.handle_message(body)
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            print("✅ Task completed and deleted from queue")
        except Exception as e:
            print(f"❌ Task failed: {e}")
            traceback.print_exc()
            self.move_to_dlq(body)

    def poll_and_process(self):
        print("📥 Starting batch polling...")
        while True:
            try:
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=20,
                    VisibilityTimeout=300
                )

                messages = response.get("Messages", [])
                if not messages:
                    print("⏳ No messages. Sleeping 10s...")
                    time.sleep(10)
                    continue

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(self._process_single, m) for m in messages]
                    for future in as_completed(futures):
                        _ = future.result()

            except KeyboardInterrupt:
                print("👋 Exiting worker loop")
                break
            except Exception as e:
                print(f"❌ Worker error: {e}")
                traceback.print_exc()
                time.sleep(5)


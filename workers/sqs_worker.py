import os
import json
import sys
from pathlib import Path

import boto3
import time
import traceback
from dotenv import load_dotenv
from botocore.exceptions import ClientError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from preflop.generate_ranges import generate_single_range

# Load AWS credentials and config
load_dotenv()

REGION = os.getenv("AWS_REGION")
QUEUE_URL = os.getenv("AWS_SQS_QUEUE_URL")
DLQ_URL = os.getenv("AWS_SQS_DLQ_URL")  # Optional fallback

sqs = boto3.client("sqs", region_name=REGION)


def move_to_dlq(message_body):
    if not DLQ_URL:
        print("⚠️ DLQ URL not set. Cannot forward failed message.")
        return
    try:
        sqs.send_message(
            QueueUrl=DLQ_URL,
            MessageBody=message_body
        )
        print("➡️ Moved failed task to DLQ")
    except ClientError as e:
        print(f"❌ Failed to move message to DLQ: {e}")


def poll_and_process():
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=300
            )

            messages = response.get("Messages", [])
            if not messages:
                print("⏳ No messages found. Sleeping 10s...")
                time.sleep(10)
                continue

            for msg in messages:
                receipt_handle = msg["ReceiptHandle"]
                body = msg["Body"]
                print("🔄 Processing task...")

                try:
                    config = json.loads(body)
                    generate_single_range(config)
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=receipt_handle
                    )
                    print("✅ Task completed and deleted from queue")

                except Exception as e:
                    print(f"❌ Task failed: {e}")
                    traceback.print_exc()
                    move_to_dlq(body)

        except KeyboardInterrupt:
            print("👋 Exiting worker loop")
            break
        except Exception as e:
            print(f"❌ Worker error: {e}")
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    poll_and_process()

import os
import json
import sys
import time
import traceback
import argparse
from pathlib import Path
import boto3

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from preflop.generate_ranges import generate_single_range


def handle_preflop_task(message_body: str):
    try:
        config = json.loads(message_body)
        generate_single_range(config)
        print("✅ Processed preflop task")
        return True
    except Exception as e:
        print(f"❌ Failed to process task: {e}")
        traceback.print_exc()
        return False


def main(dlq_url: str, region: str):
    sqs = boto3.client("sqs", region_name=region)

    print("📥 Starting DLQ Worker (single-threaded)...")
    print(f"🔗 DLQ URL: {dlq_url}")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=dlq_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=300
            )

            messages = response.get("Messages", [])
            if not messages:
                print("⏳ No messages in DLQ. Waiting...")
                time.sleep(10)
                continue

            msg = messages[0]
            receipt_handle = msg.get("ReceiptHandle")
            body = msg.get("Body")
            message_id = msg.get("MessageId")

            print(f"📩 Processing DLQ Message ID: {message_id}")
            success = handle_preflop_task(body)

            if success:
                sqs.delete_message(QueueUrl=dlq_url, ReceiptHandle=receipt_handle)
                print(f"🧹 Deleted DLQ message: {message_id}\n")
            else:
                print(f"⏩ Skipping deletion. Will retry DLQ message: {message_id}\n")

        except KeyboardInterrupt:
            print("👋 DLQ Worker interrupted. Exiting...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlq-url", default=os.getenv("AWS_SQS_DLQ_URL"), help="SQS DLQ URL")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"), help="AWS Region")
    args = parser.parse_args()

    if not args.dlq_url:
        raise SystemExit("❌ DLQ URL must be provided via --dlq-url or AWS_SQS_DLQ_URL env var")

    main(dlq_url=args.dlq_url, region=args.region)
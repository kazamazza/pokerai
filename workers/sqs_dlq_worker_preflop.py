import os
import json
import time
import traceback
import boto3
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

REGION = os.getenv("AWS_REGION")
DLQ_URL = os.getenv("AWS_SQS_DLQ_URL")
MAX_RETRIES = int(os.getenv("DLQ_MAX_RETRIES", 3))

sqs = boto3.client("sqs", region_name=REGION)

def process_message(config):
    # TODO: Add your retry-safe processing logic here
    # from preflop.generate_ranges import generate_single_range
    print("🔄 Processing DLQ message:", config)
    # generate_single_range(config)  # re-attempt the same logic

def main():
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=DLQ_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=300
            )

            messages = response.get("Messages", [])
            if not messages:
                print("⏳ No DLQ messages. Sleeping 10s...")
                time.sleep(10)
                continue

            for msg in messages:
                receipt_handle = msg["ReceiptHandle"]
                body = msg["Body"]
                retries = int(msg.get("Attributes", {}).get("ApproximateReceiveCount", 1))

                try:
                    config = json.loads(body)
                    print(f"🔁 Retry attempt {retries}/{MAX_RETRIES}")

                    if retries > MAX_RETRIES:
                        print("🚫 Max retries exceeded. Skipping message.")
                        continue

                    process_message(config)

                    sqs.delete_message(
                        QueueUrl=DLQ_URL,
                        ReceiptHandle=receipt_handle
                    )
                    print("✅ Message deleted from DLQ")

                except Exception as e:
                    print(f"❌ Error processing DLQ message: {e}")
                    traceback.print_exc()

        except KeyboardInterrupt:
            print("👋 Exiting DLQ worker loop")
            break
        except Exception as e:
            print(f"❌ Worker error: {e}")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()
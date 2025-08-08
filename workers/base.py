import os
import time
import traceback
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import boto3
from botocore.exceptions import ClientError

class SQSWorker:
    def __init__(
        self,
        handler: Callable[[str], None],
        max_threads: int = 5,
        batch_size: int = 5,
        region: str | None = None,
        queue_url: str | None = None,
        dlq_url: str | None = None
    ):
        self.handler = handler
        self.batch_size = batch_size
        self.max_threads = max_threads

        self.region = region or os.environ.get("AWS_REGION")
        self.queue_url = queue_url or os.environ.get("AWS_SQS_QUEUE_URL")
        self.dlq_url = dlq_url or os.environ.get("AWS_SQS_DLQ_URL")
        if not self.region or not self.queue_url:
            raise RuntimeError("Missing AWS_REGION or AWS_SQS_QUEUE_URL")

        self.sqs = boto3.client("sqs", region_name=self.region)
        self.ec2 = boto3.client("ec2", region_name=self.region)

    def move_to_dlq(self, message_body: str):
        if not self.dlq_url:
            print("⚠️ DLQ URL not set. Cannot forward failed message.")
            return
        try:
            self.sqs.send_message(QueueUrl=self.dlq_url, MessageBody=message_body)
            print("➡️ Moved failed task to DLQ")
        except ClientError as e:
            print(f"❌ Failed to move message to DLQ: {e}")

    def _process_single(self, msg):
        rh = msg.get("ReceiptHandle")
        body = msg.get("Body")
        mid = msg.get("MessageId")
        if not rh or not body:
            print(f"⚠️ Skipping invalid message: {msg}")
            return

        # heartbeat to keep message invisible while computing
        stop = False
        def extender():
            while not stop:
                try:
                    self.sqs.change_message_visibility(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=rh,
                        VisibilityTimeout=600  # 10 min window
                    )
                    print(f"[vis] extended {mid} to 600s")
                except Exception as e:
                    print(f"[vis] extend failed for {mid}: {e}")
                time.sleep(60 + random.randint(0, 15))
        hb = threading.Thread(target=extender, daemon=True)
        hb.start()

        try:
            start = time.time()
            print(f"📩 Received {mid}")
            self.handler(body)
            dur = time.time() - start
            print(f"✅ Task {mid} completed in {dur:.2f}s")

            # robust delete with retry
            for attempt in range(3):
                try:
                    self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=rh)
                    print(f"🧹 Deleted {mid}")
                    break
                except Exception as e:
                    print(f"⚠️ Delete failed for {mid} (try {attempt+1}/3): {e}")
                    time.sleep(2 ** attempt)
            else:
                print(f"❌ Could not delete {mid}; sending to DLQ")
                self.move_to_dlq(body)
        except Exception as e:
            print(f"❌ Task failed for {mid}: {e}")
            traceback.print_exc()
            self.move_to_dlq(body)
        finally:
            stop = True

    def run(self):
        print("📥 Starting SQS polling loop...")
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            while True:
                try:
                    try:
                        resp = self.sqs.receive_message(
                            QueueUrl=self.queue_url,
                            MaxNumberOfMessages=self.batch_size,
                            WaitTimeSeconds=20,
                            VisibilityTimeout=60   # short initial; we extend immediately
                        )
                    except ClientError as e:
                        print(f"❌ receive_message failed: {e}")
                        time.sleep(5)
                        continue

                    messages = resp.get("Messages", [])
                    if not messages:
                        print("⏳ No messages. Checking shutdown condition...")
                        if self._should_shutdown():
                            print("✅ Queue empty; shutting down.")
                            self._shutdown_instance()
                            break
                        time.sleep(10)
                        continue

                    print(f"📦 Received {len(messages)} messages")
                    # pre-extend each while we spin up work
                    for m in messages:
                        try:
                            self.sqs.change_message_visibility(
                                QueueUrl=self.queue_url,
                                ReceiptHandle=m["ReceiptHandle"],
                                VisibilityTimeout=600
                            )
                            print(f"[vis] pre-extended {m['MessageId']} to 600s")
                        except Exception as e:
                            print(f"[vis] pre-extend failed for {m.get('MessageId')}: {e}")

                    batch_start = time.time()
                    futures = [executor.submit(self._process_single, m) for m in messages]
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"❌ Error during task execution: {e}")
                            traceback.print_exc()
                    print(f"⏱️ Finished batch of {len(messages)} in {time.time() - batch_start:.2f}s\n")

                except KeyboardInterrupt:
                    print("👋 Exiting worker loop")
                    break
                except Exception as e:
                    print(f"❌ Worker error: {e}")
                    traceback.print_exc()
                    time.sleep(5)

    def _should_shutdown(self):
        try:
            attrs = self.sqs.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=[
                    "ApproximateNumberOfMessages",
                    "ApproximateNumberOfMessagesNotVisible"
                ]
            )["Attributes"]
            visible = int(attrs.get("ApproximateNumberOfMessages", 0))
            inflight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
            return visible == 0 and inflight == 0
        except Exception as e:
            print(f"❌ Failed to check queue for shutdown: {e}")
            return False

    def _shutdown_instance(self):
        try:
            import requests
            iid = requests.get("http://169.254.169.254/latest/meta-data/instance-id", timeout=2).text
            self.ec2.stop_instances(InstanceIds=[iid])
            print(f"🛑 Shutting down instance: {iid}")
        except Exception as e:
            print(f"❌ Failed to shutdown instance: {e}")
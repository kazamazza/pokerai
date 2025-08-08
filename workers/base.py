import os
import threading
import time
import traceback
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
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
            self.sqs.send_message(
                QueueUrl=self.dlq_url,
                MessageBody=message_body
            )
            print("➡️ Moved failed task to DLQ")
        except ClientError as e:
            print(f"❌ Failed to move message to DLQ: {e}")

    def _process_single(self, msg):
        receipt_handle = msg.get("ReceiptHandle")
        body = msg.get("Body")
        message_id = msg.get("MessageId")

        if not receipt_handle or not body:
            print(f"⚠️ Skipping invalid message: {msg}")
            return

        # --- keep-alive thread to extend visibility while we work ---
        stop_flag = False

        def _extend_loop():
            while not stop_flag:
                try:
                    self.sqs.change_message_visibility(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=receipt_handle,
                        VisibilityTimeout=300  # extend by 5 min
                    )
                except Exception as e:
                    print(f"⚠️ Visibility extend failed: {e}")
                time.sleep(90)  # heartbeat

        hb = threading.Thread(target=_extend_loop, daemon=True)
        hb.start()
        # ------------------------------------------------------------

        try:
            start = time.time()
            print(f"📩 Received Message ID: {message_id}")
            print("🔄 Processing task...")
            self.handler(body)
            print(f"✅ Task completed in {time.time() - start:.2f}s")

            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
            print(f"🧹 Deleted Message ID: {message_id}")
        except Exception as e:
            print(f"❌ Task failed: {e}")
            traceback.print_exc()
            self.move_to_dlq(body)
        finally:
            stop_flag = True
            # let the heartbeat exit; don't block long
            try:
                hb.join(timeout=1)
            except Exception:
                pass

    def run(self):
        print("\U0001F4E5 Starting SQS polling loop...")

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            while True:
                try:
                    try:
                        response = self.sqs.receive_message(
                            QueueUrl=self.queue_url,
                            MaxNumberOfMessages=self.batch_size,
                            WaitTimeSeconds=20,
                            VisibilityTimeout=60  # short initial; we'll extend immediately per message
                        )
                    except ClientError as e:
                        print(f"❌ SQS receive_message failed: {e}")
                        time.sleep(5)
                        continue

                    messages = response.get("Messages", [])
                    if not messages:
                        print("⏳ No messages. Checking shutdown condition...")
                        if self._should_shutdown():
                            print("✅ Queue is empty and idle. Initiating shutdown.")
                            self._shutdown_instance()
                            break
                        time.sleep(10)
                        continue

                    print(f"📦 Received {len(messages)} messages")
                    # Pre-extend each message so they don't reappear while we queue work
                    for m in messages:
                        try:
                            self.sqs.change_message_visibility(
                                QueueUrl=self.queue_url,
                                ReceiptHandle=m["ReceiptHandle"],
                                VisibilityTimeout=300  # 5 min runway to start
                            )
                        except Exception as e:
                            print(f"⚠️ Pre-extend failed: {e}")

                    batch_start = time.time()

                    futures = [executor.submit(self._process_single, m) for m in messages]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except TypeError as e:
                            print(f"🚨 TypeError during task: {e}")
                            traceback.print_exc()
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
            response = self.sqs.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
            )
            visible = int(response["Attributes"].get("ApproximateNumberOfMessages", 0))
            inflight = int(response["Attributes"].get("ApproximateNumberOfMessagesNotVisible", 0))
            return visible == 0 and inflight == 0
        except Exception as e:
            print(f"❌ Failed to check queue for shutdown condition: {e}")
            return False

    def _shutdown_instance(self):
        try:
            import requests
            r = requests.get("http://169.254.169.254/latest/meta-data/instance-id", timeout=2)
            instance_id = r.text
            self.ec2.stop_instances(InstanceIds=[instance_id])
            print(f"🛑 Shutting down instance: {instance_id}")
        except Exception as e:
            print(f"❌ Failed to shutdown instance: {e}")
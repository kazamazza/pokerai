#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import ClientError


def get_queue_attrs(sqs, queue_url: str, names: List[str]) -> Dict[str, str]:
    return sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=names)["Attributes"]


def try_start_move_task(sqs, main_url: str, dlq_url: str) -> Optional[str]:
    """Attempt fast server-side redrive. Returns task ARN if started, else None."""
    try:
        main_arn = get_queue_attrs(sqs, main_url, ["QueueArn"])["QueueArn"]
        resp = sqs.start_message_move_task(QueueUrl=dlq_url, SourceArn=main_arn)
        return resp.get("TaskHandle")
    except ClientError as e:
        # Unsupported in some regions/accounts or permission missing—fall back to manual
        print(f"[info] start_message_move_task not used: {e.response.get('Error', {}).get('Code')}", flush=True)
        return None


def wait_move_task(sqs, dlq_url: str, task_handle: str, poll=5):
    """Polls the move task until it finishes."""
    print(f"[info] Move task started: {task_handle}")
    while True:
        resp = sqs.list_message_move_tasks(QueueUrl=dlq_url, MaxResults=10)
        items = resp.get("Results", [])
        match = next((x for x in items if x.get("TaskHandle") == task_handle), None)
        if match:
            status = match.get("Status")
            print(f"[info] task status: {status}")
            if status in ("COMPLETED", "CANCELED", "FAILED"):
                return status
        time.sleep(poll)


def manual_redrive(
    sqs,
    main_url: str,
    dlq_url: str,
    batch_size: int = 10,
    max_messages: Optional[int] = None,
    preserve_attrs: bool = True,
    dry_run: bool = False,
):
    moved = 0
    backoff = 1

    while True:
        if max_messages is not None and moved >= max_messages:
            break

        try:
            resp = sqs.receive_message(
                QueueUrl=dlq_url,
                MaxNumberOfMessages=min(batch_size, (max_messages - moved) if max_messages else batch_size),
                WaitTimeSeconds=2,
                VisibilityTimeout=60,
                AttributeNames=["All"],           # includes FIFO attributes if present
                MessageAttributeNames=["All"] if preserve_attrs else [],
            )
        except ClientError as e:
            print(f"[warn] receive_message failed: {e}", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        msgs = resp.get("Messages", [])
        if not msgs:
            break

        entries_delete = []
        for m in msgs:
            body = m["Body"]
            msg_attrs = m.get("MessageAttributes") or {}
            attrs = m.get("Attributes") or {}

            send_kwargs: Dict[str, Any] = {
                "QueueUrl": main_url,
                "MessageBody": body,
            }
            if preserve_attrs and msg_attrs:
                send_kwargs["MessageAttributes"] = msg_attrs

            # FIFO handling
            # We detect FIFO from queue attribute, but we can also look at MessageGroupId presence.
            # Safer: check main queue attr once outside loop.
            # For simplicity here, we infer from Attributes if provided.
            group_id = attrs.get("MessageGroupId")
            dedup_id = attrs.get("MessageDeduplicationId")
            if group_id:
                send_kwargs["MessageGroupId"] = group_id
            if dedup_id:
                send_kwargs["MessageDeduplicationId"] = dedup_id

            if dry_run:
                ok = True
            else:
                try:
                    sqs.send_message(**send_kwargs)
                    ok = True
                except ClientError as e:
                    print(f"[error] send_message failed: {e}", flush=True)
                    ok = False

            if ok:
                entries_delete.append(
                    {"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]}
                )
                moved += 1

        if entries_delete and not dry_run:
            # Best-effort batch delete; if it fails, those messages may reappear later.
            try:
                sqs.delete_message_batch(QueueUrl=dlq_url, Entries=entries_delete)
            except ClientError as e:
                print(f"[warn] delete_message_batch failed: {e}", flush=True)

        print(f"[info] moved so far: {moved}", flush=True)

    print(f"[done] total moved: {moved}")


def main():
    ap = argparse.ArgumentParser(description="Redrive SQS DLQ back to main queue")
    ap.add_argument("--region", default=None, help="AWS region (defaults to env/SDK config)")
    ap.add_argument("--main-queue", required=True, help="Main queue URL")
    ap.add_argument("--dlq-queue", required=True, help="DLQ queue URL")
    ap.add_argument("--manual", action="store_true", help="Force manual redrive (skip StartMessageMoveTask)")
    ap.add_argument("--max", type=int, default=None, help="Max messages to move (manual mode)")
    ap.add_argument("--batch-size", type=int, default=10, help="Manual receive/send batch size (<=10)")
    ap.add_argument("--no-attrs", action="store_true", help="Do not preserve MessageAttributes on resend")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not send/delete")
    args = ap.parse_args()

    sqs = boto3.client("sqs", region_name=args.region)

    # Quick stats
    try:
        dlq_attrs = get_queue_attrs(
            sqs,
            args.dlq_queue,
            ["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible", "FifoQueue"],
        )
    except ClientError as e:
        print(f"[fatal] cannot read DLQ attributes: {e}", file=sys.stderr)
        sys.exit(2)

    dlq_visible = int(dlq_attrs.get("ApproximateNumberOfMessages", 0))
    dlq_inflight = int(dlq_attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
    is_fifo = dlq_attrs.get("FifoQueue", "false").lower() == "true"
    print(
        f"[info] DLQ visible={dlq_visible} inflight={dlq_inflight} fifo={is_fifo} "
        f"(dry_run={args.dry_run}, manual={args.manual})"
    )

    if not args.manual and not args.dry_run:
        task_handle = try_start_move_task(sqs, args.main_queue, args.dlq_queue)
        if task_handle:
            status = wait_move_task(sqs, args.dlq_queue, task_handle)
            print(f"[done] redrive task completed with status: {status}")
            return

    # Manual redrive
    manual_redrive(
        sqs,
        args.main_queue,
        args.dlq_queue,
        batch_size=min(max(args.batch_size, 1), 10),
        max_messages=args.max,
        preserve_attrs=not args.no_attrs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
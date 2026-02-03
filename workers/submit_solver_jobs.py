from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


import json
from typing import Any, Dict, Iterable, List

import boto3
import pandas as pd

from workers.solver_job_schema import (
    build_messages,
    validate_manifest_columns,
)


def submit_batches(
    *,
    sqs,
    queue_url: str,
    messages: Iterable[Dict[str, Any]],
    batch_size: int,
) -> int:
    """
    Submit messages to SQS in batches (max 10).
    """
    sent = 0
    batch: List[Dict[str, Any]] = []

    def flush(entries: List[Dict[str, Any]]) -> None:
        nonlocal sent
        if not entries:
            return

        resp = sqs.send_message_batch(
            QueueUrl=queue_url,
            Entries=entries,
        )

        failed = resp.get("Failed", []) or []
        if failed:
            raise RuntimeError(f"SQS batch failed: {failed}")

        sent += len(resp.get("Successful", []) or [])

    for i, msg in enumerate(messages):
        # prefer stable id if present; else fall back to sha1/size_pct/i
        job_id = msg.get("job_id")
        if isinstance(job_id, str) and job_id:
            entry_id = job_id.replace(":", "_")
        else:
            sha1 = msg.get("sha1", "unknown")
            size_pct = ((msg.get("params") or {}).get("size_pct")) if isinstance(msg.get("params"), dict) else None
            entry_id = f"{sha1}_{size_pct}_{i}"

        entry = {
            "Id": entry_id[:80],  # SQS Id max len is 80
            "MessageBody": json.dumps(msg, separators=(",", ":")),
        }

        batch.append(entry)
        if len(batch) >= batch_size:
            flush(batch)
            batch.clear()

    if batch:
        flush(batch)

    return sent


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest parquet file")
    ap.add_argument("--queue-url", required=True, help="SQS queue URL")
    ap.add_argument("--batch-size", type=int, default=10, help="SQS batch size (1..10)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of manifest rows")
    ap.add_argument("--dry-run", action="store_true", help="Do not submit; just print sample")
    args = ap.parse_args()

    if not (1 <= args.batch_size <= 10):
        raise SystemExit("batch-size must be 1..10")

    df = pd.read_parquet(args.manifest)

    ok, missing = validate_manifest_columns(df.columns)
    if not ok:
        raise SystemExit(f"Manifest missing columns: {missing}")

    if args.limit:
        df = df.head(args.limit)

    rows = list(df.itertuples(index=False))

    all_messages: List[Dict[str, Any]] = []
    for row in rows:
        all_messages.extend(build_messages(row))

    if not all_messages:
        print("⚠️ No messages produced (empty manifest after filters).")
        return

    if args.dry_run:
        print(f"🧪 dry-run: would submit {len(all_messages)} messages")
        print(json.dumps(all_messages[0], indent=2))
        return

    sqs = boto3.client("sqs")
    sent = submit_batches(
        sqs=sqs,
        queue_url=args.queue_url,
        messages=all_messages,
        batch_size=args.batch_size,
    )

    print(f"✅ submitted {sent} solver jobs")


if __name__ == "__main__":
    main()
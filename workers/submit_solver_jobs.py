from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List
import pandas as pd
import boto3


# =========================
# Parsing helpers (local)
# =========================

def parse_bet_sizes(cell) -> List[int]:
    """
    Parse manifest bet_sizes into ordered unique integer percents.
    Accepts:
      - [0.33, 0.66]
      - [33, 66]
      - pyarrow list structs: [{"element": 0.33}, ...]
    """
    if cell is None:
        return []

    # unwrap pyarrow / numpy
    try:
        import pyarrow as pa
        if hasattr(cell, "as_py"):
            cell = cell.as_py()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(cell, np.ndarray):
            cell = cell.tolist()
    except Exception:
        pass

    seq = cell if isinstance(cell, list) else [cell]
    out: List[int] = []
    seen = set()

    for it in seq:
        if it is None:
            continue
        v = it.get("element") if isinstance(it, dict) else it
        try:
            f = float(v)
        except Exception:
            continue

        if f <= 3.0:
            pct = int(Decimal(str(f * 100)).quantize(0, ROUND_HALF_UP))
        else:
            pct = int(Decimal(str(f)).quantize(0, ROUND_HALF_UP))

        if 1 <= pct <= 200 and pct not in seen:
            out.append(pct)
            seen.add(pct)

    return out


def inject_size_into_s3_key(base_key: str, size_pct: int) -> str:
    base = base_key.rstrip("/")
    return f"{base}/size={int(size_pct)}p/output_result.json.gz"


# =========================
# Submitter
# =========================

REQUIRED_COLUMNS = {
    "sha1",
    "s3_key",
    "street",
    "pot_bb",
    "effective_stack_bb",
    "ip_pos",
    "oop_pos",
    "board",
    "board_cluster_id",
    "bet_sizing_id",
    "bet_sizes",
    "range_ip",
    "range_oop",
    "accuracy",
    "max_iter",
    "allin_threshold",
    "solver_version",
    "node_key",
}


PARAM_FIELDS = [
    "street",
    "pot_bb",
    "effective_stack_bb",
    "ip_pos",
    "oop_pos",
    "board",
    "board_cluster_id",
    "bet_sizing_id",
    "range_ip",
    "range_oop",
    "accuracy",
    "max_iter",
    "allin_threshold",
    "solver_version",
    "node_key",
]


def build_messages(row) -> List[Dict[str, Any]]:
    sizes = parse_bet_sizes(row.bet_sizes)
    if not sizes:
        raise ValueError(f"Manifest row sha1={row.sha1} has no bet_sizes")

    messages = []

    for size_pct in sizes:
        params = {k: getattr(row, k) for k in PARAM_FIELDS}
        params["size_pct"] = int(size_pct)

        msg = {
            "sha1": row.sha1,
            "s3_key": inject_size_into_s3_key(row.s3_key, size_pct),
            "params": params,
        }

        messages.append(msg)

    return messages


def submit_batches(
    *,
    sqs,
    queue_url: str,
    messages: Iterable[Dict[str, Any]],
    batch_size: int,
) -> int:
    sent = 0
    batch: List[Dict[str, Any]] = []

    def flush(entries):
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
        sent += len(resp.get("Successful", []))

    for i, msg in enumerate(messages):
        entry = {
            "Id": f"{msg['sha1']}_{msg['params']['size_pct']}_{i}",
            "MessageBody": json.dumps(msg),
        }
        batch.append(entry)
        if len(batch) == batch_size:
            flush(batch)
            batch.clear()

    if batch:
        flush(batch)

    return sent


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--queue-url", required=True)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (1 <= args.batch_size <= 10):
        raise SystemExit("batch-size must be 1..10")

    df = pd.read_parquet(args.manifest)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")

    if args.limit:
        df = df.head(args.limit)

    rows = list(df.itertuples(index=False))

    all_messages: List[Dict[str, Any]] = []
    for row in rows:
        all_messages.extend(build_messages(row))

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
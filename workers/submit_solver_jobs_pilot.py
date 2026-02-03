# workers/submit_solver_jobs_pilot.py
from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import argparse
from workers.solver_job_schema import PARAM_FIELDS, validate_manifest_columns

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

    # unwrap pyarrow
    try:
        if hasattr(cell, "as_py"):
            cell = cell.as_py()
    except Exception:
        pass

    # unwrap numpy
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
    base = str(base_key).rstrip("/")
    return f"{base}/size={int(size_pct)}p/output_result.json.gz"


def build_messages(row) -> List[Dict[str, Any]]:
    sizes = parse_bet_sizes(row.bet_sizes)
    if not sizes:
        raise ValueError(f"Manifest row sha1={row.sha1} has no bet_sizes")

    messages: List[Dict[str, Any]] = []
    for size_pct in sizes:
        params = {k: getattr(row, k) for k in PARAM_FIELDS}
        params["size_pct"] = int(size_pct)

        msg = {
            "sha1": str(row.sha1),
            "s3_key": inject_size_into_s3_key(row.s3_key, size_pct),
            "params": params,
        }
        messages.append(msg)

    return messages


# ----------------------------
# SQS submit
# ----------------------------

def submit_batches(
    *,
    sqs,
    queue_url: str,
    messages: Iterable[Dict[str, Any]],
    batch_size: int,
) -> int:
    sent = 0
    batch: List[Dict[str, Any]] = []

    def flush(entries: List[Dict[str, Any]]) -> None:
        nonlocal sent
        if not entries:
            return
        resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        failed = resp.get("Failed", []) or []
        if failed:
            raise RuntimeError(f"SQS batch failed: {failed}")
        sent += len(resp.get("Successful", []) or [])

    for i, msg in enumerate(messages):
        sha1 = msg.get("sha1", "unknown")
        size_pct = ((msg.get("params") or {}).get("size_pct")) if isinstance(msg.get("params"), dict) else None
        entry_id = f"{sha1}_{size_pct}_{i}"
        batch.append(
            {
                "Id": entry_id[:80],
                "MessageBody": json.dumps(msg, separators=(",", ":")),
            }
        )
        if len(batch) >= batch_size:
            flush(batch)
            batch.clear()

    if batch:
        flush(batch)

    return sent


# ----------------------------
# Pilot selection
# ----------------------------

def _bet_size_count(cell) -> int:
    try:
        return len(parse_bet_sizes(cell))
    except Exception:
        return 0


def pick_one_per_scenario_family(
    df: pd.DataFrame,
    *,
    streets: Sequence[int],
    per_group: int = 1,
) -> pd.DataFrame:
    """
    Pick exactly N rows per (street, bet_sizing_id).

    Within each group, prefer the "hardest-looking" row:
      1) most bet sizes
      2) higher effective_stack_bb
      3) higher pot_bb
    """
    work = df.copy()

    # filter to requested streets
    work = work[work["street"].isin(list(streets))].copy()
    if work.empty:
        return work

    # compute bet_size_count for "hardness"
    work["_bet_size_count"] = work["bet_sizes"].apply(_bet_size_count)

    # sort so groupby().head(per_group) yields "hard" samples deterministically
    sort_cols = ["_bet_size_count", "effective_stack_bb", "pot_bb"]
    for c in sort_cols:
        if c not in work.columns:
            # shouldn't happen given REQUIRED_COLUMNS, but keep safe
            work[c] = 0
    work = work.sort_values(sort_cols, ascending=[False, False, False])

    grp = work.groupby(["street", "bet_sizing_id"], dropna=False)
    out = grp.head(per_group).reset_index(drop=True)
    return out.drop(columns=["_bet_size_count"], errors="ignore")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("Pilot submitter: 1 row per (street, bet_sizing_id)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--queue-url", required=True)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--streets", default="1,2,3", help="Comma streets (0=pre,1=flop,2=turn,3=river). Default 1,2,3.")
    ap.add_argument("--per-group", type=int, default=1, help="Rows per (street, bet_sizing_id). Default 1.")
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit manifest rows before selection")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (1 <= args.batch_size <= 10):
        raise SystemExit("batch-size must be 1..10")

    streets = [int(x.strip()) for x in args.streets.split(",") if x.strip() != ""]
    if not streets:
        raise SystemExit("--streets must include at least one street id")

    df = pd.read_parquet(args.manifest)
    ok, missing = validate_manifest_columns(df.columns)
    if not ok:
        raise SystemExit(f"Manifest missing columns: {missing}")

    if args.limit:
        df = df.head(args.limit)

    pilot_df = pick_one_per_scenario_family(
        df,
        streets=streets,
        per_group=max(1, int(args.per_group)),
    )

    n_groups = pilot_df.groupby(["street", "bet_sizing_id"], dropna=False).ngroups if len(pilot_df) else 0
    print(f"✅ pilot picked {len(pilot_df)} manifest rows across {n_groups} (street, bet_sizing_id) groups")

    # Expand into solver messages (per bet size)
    rows = list(pilot_df.itertuples(index=False))
    all_messages: List[Dict[str, Any]] = []
    for row in rows:
        all_messages.extend(build_messages(row))

    print(f"✅ expanded to {len(all_messages)} solver messages (after bet_sizes expansion)")

    if not all_messages:
        print("⚠️ No messages produced.")
        return

    if args.dry_run:
        print("🧪 dry-run sample message:")
        print(json.dumps(all_messages[0], indent=2))
        print("🧪 groups included:")
        preview = pilot_df[["street", "bet_sizing_id", "effective_stack_bb", "pot_bb"]].head(50)
        print(preview.to_string(index=False))
        return

    sqs = boto3.client("sqs")
    sent = submit_batches(
        sqs=sqs,
        queue_url=args.queue_url,
        messages=all_messages,
        batch_size=int(args.batch_size),
    )
    print(f"🚀 submitted {sent} pilot solver jobs")


if __name__ == "__main__":
    main()
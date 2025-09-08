import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

def main():
    import argparse
    import time
    import json
    import os
    import pandas as pd
    import boto3
    import botocore

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--queue_url", required=True)
    ap.add_argument("--batch", type=int, default=10, help="SQS batch size (≤10)")
    ap.add_argument("--limit", type=int, default=None, help="Only send first N jobs")
    ap.add_argument("--dry-run", action="store_true", help="Don’t send, just show counts")
    args = ap.parse_args()

    if args.batch < 1 or args.batch > 10:
        raise SystemExit("--batch must be between 1 and 10 (SQS limit)")

    df = pd.read_parquet(args.manifest)
    # Enforce limit early
    if args.limit is not None:
        df = df.head(int(args.limit))

    # Required manifest columns (some are optional with defaults)
    # Columns always needed at the envelope level:
    ENVELOPE_KEYS = ["sha1", "s3_key"]

    # Parameters we *may* include if present in the manifest.
    # FLOP-only fields (e.g., board_cluster_id) are optional.
    PARAM_WHITELIST = [
        # common
        "street", "pot_bb", "effective_stack_bb", "positions", "bet_sizing_id",
        "range_ip", "range_oop",
        "accuracy", "max_iter", "allin_threshold",
        "node_key", "solver_version",

        # board state
        "board",  # "QsJh2h" (flop) or "QsJh2h7d" (turn) or "QsJh2h7d2s" (river)
        "board_cluster_id",  # frequently flop-only (OK if missing)

        # turn/river / tree linkage (optional; include if your manifests add them)
        "parent_sha1", "parent_node_key", "line_key",
        "street_root_state",  # e.g. compressed state id if you use one
        "turn_card", "river_card",  # if you choose to split cards instead of whole board
    ]

    missing_env = [k for k in ENVELOPE_KEYS if k not in df.columns]
    if missing_env:
        raise SystemExit(f"Manifest missing required columns: {missing_env}")

    # Sensible defaults to keep flop manifests happy
    if "street" not in df.columns:
        df = df.assign(street=1)
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")
    if "solver_version" not in df.columns:
        df = df.assign(solver_version="v1")

    def to_msg(row):
        params = {}
        for k in PARAM_WHITELIST:
            if k in row and pd.notna(row[k]):
                v = row[k]
                if k in ("street", "max_iter", "board_cluster_id"):
                    try:
                        v = int(v)
                    except:
                        continue
                elif k in ("pot_bb", "effective_stack_bb", "accuracy", "allin_threshold"):
                    try:
                        v = float(v)
                    except:
                        continue
                elif isinstance(v, (int, float, str)):
                    pass
                else:
                    v = str(v)
                params[k] = v

        msg = {
            "sha1": str(row["sha1"]),
            "s3_key": str(row["s3_key"]),
            "params": params,
        }
        return {"Id": str(row["sha1"])[:80], "MessageBody": json.dumps(msg)}

    if args.dry_run:
        print("🧪 dry-run: not sending to SQS")
        print(df.head(min(3, len(df))).to_string(index=False))
        return

    sqs = boto3.client(
        "sqs",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    n = len(df)

    i = 0
    sent = 0
    retries = 0

    while i < n:
        chunk = df.iloc[i:i + args.batch]
        entries = [to_msg(r) for _, r in chunk.iterrows()]

        # always define resp2 so we can sum safely
        resp2 = {"Successful": []}

        try:
            resp = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=entries)
            failed = resp.get("Failed", []) or []

            if failed:
                # retry once only failed entries
                fail_ids = {f["Id"] for f in failed}
                retry_entries = [e for e in entries if e["Id"] in fail_ids]
                if retry_entries:
                    time.sleep(0.5)
                    retries += 1
                    resp2 = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=retry_entries)
                    failed2 = resp2.get("Failed", []) or []
                    if failed2:
                        # still failed after retry — log a short preview and continue
                        print(f"⚠️  batch @ {i}: {len(failed2)} entries failed after retry "
                              f"(e.g. {failed2[0] if failed2 else ''})")

            ok_count = len(resp.get("Successful", [])) + len(resp2.get("Successful", []))
            sent += ok_count

        except botocore.exceptions.BotoCoreError as e:
            print(f"⚠️  SQS error at batch starting {i}: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error at batch starting {i}: {e}")

        i += args.batch
        if sent and (sent % (args.batch * 20) == 0 or i >= n):
            print(f"… progress: sent≈{sent}/{n}")

    print(f"✅ submitted ~{sent}/{n} jobs to SQS  (retries: {retries})")

if __name__ == "__main__":
    main()
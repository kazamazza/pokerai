import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_model_config

def main():
    import argparse, time, json, os
    import pandas as pd
    import boto3, botocore

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--queue_url", required=True)
    ap.add_argument("--batch", type=int, default=10, help="SQS batch size (≤10)")
    ap.add_argument("--limit", type=int, default=None, help="Only send first N jobs")
    ap.add_argument("--dry-run", action="store_true", help="Don’t send, just show counts")
    ap.add_argument("--config", default=None, help="Optional YAML to fill *missing* non-critical fields")
    args = ap.parse_args()

    # Optional config (only for filler defaults, never to override manifest)
    solver_cfg = {}
    if args.config:
        try:
            cfg = load_model_config(args.config) or {}
            solver_cfg = (cfg.get("solver") or cfg.get("worker") or {}) or {}
        except Exception:
            solver_cfg = {}

    if not (1 <= args.batch <= 10):
        raise SystemExit("--batch must be between 1 and 10 (SQS limit)")

    df = pd.read_parquet(args.manifest)
    if args.limit is not None:
        df = df.head(int(args.limit))

    ENVELOPE_KEYS = ["sha1", "s3_key"]
    missing_env = [k for k in ENVELOPE_KEYS if k not in df.columns]
    if missing_env:
        raise SystemExit(f"Manifest missing required columns: {missing_env}")

    # Safe defaults for missing lightweight fields
    if "street" not in df.columns:
        df = df.assign(street=1)
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")
    if "solver_version" not in df.columns:
        df = df.assign(solver_version=solver_cfg.get("version", "v1"))
    # If threads is desired and not present in manifest, attach as column
    if "threads" not in df.columns and "threads" in solver_cfg:
        df = df.assign(threads=int(solver_cfg["threads"]))

    # Only copy what the worker understands; do NOT override manifest values
    PARAM_WHITELIST = [
        "street", "pot_bb", "effective_stack_bb", "positions", "bet_sizing_id",
        "range_ip", "range_oop",
        "accuracy", "max_iter", "allin_threshold",
        "node_key", "solver_version", "threads",
        "board", "board_cluster_id",
        "parent_sha1", "parent_node_key", "line_key",
        "street_root_state", "turn_card", "river_card",
    ]

    def _coerce_param(k, v):
        if v is None: return None
        if k in ("street", "max_iter", "board_cluster_id"):
            try: return int(v)
            except: return None
        if k in ("pot_bb", "effective_stack_bb", "accuracy", "allin_threshold"):
            try: return float(v)
            except: return None
        # keep simple scalars/strings
        return v if isinstance(v, (int, float, str)) else str(v)

    def to_msg(row):
        params = {}
        for k in PARAM_WHITELIST:
            if k in row and pd.notna(row[k]):
                v = _coerce_param(k, row[k])
                if v is not None:
                    params[k] = v

        # Never overwrite accuracy/max_iter/allin_threshold here.
        # (solver_cfg is only used above to fill missing benign fields)

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

    n, i, sent, retries = len(df), 0, 0, 0
    while i < n:
        chunk = df.iloc[i:i + args.batch]
        entries = [to_msg(r) for _, r in chunk.iterrows()]
        resp2 = {"Successful": []}
        try:
            resp = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=entries)
            failed = resp.get("Failed", []) or []
            if failed:
                fail_ids = {f["Id"] for f in failed}
                retry_entries = [e for e in entries if e["Id"] in fail_ids]
                if retry_entries:
                    time.sleep(0.5)
                    retries += 1
                    resp2 = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=retry_entries)
                    failed2 = resp2.get("Failed", []) or []
                    if failed2:
                        print(f"⚠️  batch @ {i}: {len(failed2)} entries failed after retry "
                              f"(e.g. {failed2[0] if failed2 else ''})")
            sent += len(resp.get("Successful", [])) + len(resp2.get("Successful", []))
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
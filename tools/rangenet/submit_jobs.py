import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_model_config

# --- ADD helpers somewhere above main() ---

def parse_bet_sizes_cell(cell) -> list[int]:
    """
    Manifest bet_sizes can be:
      - None
      - [0.33, 0.66]
      - [{"element": 0.33}, {"element": 0.66}]
    Return list of integer percents, e.g. [33, 66].
    """
    if cell is None:
        return []
    vals = []
    if isinstance(cell, list):
        for it in cell:
            if it is None:
                continue
            if isinstance(it, dict):
                v = it.get("element", None)
            else:
                v = it
            try:
                f = float(v)
                # if user already stored percents (e.g. 33.0), keep; else convert 0.33->33
                if f <= 3.0:
                    vals.append(int(round(100.0 * f)))
                else:
                    vals.append(int(round(f)))
            except Exception:
                pass
    else:
        # single scalar
        try:
            f = float(cell)
            vals.append(int(round(100.0 * f if f <= 3.0 else f)))
        except Exception:
            pass
    # de-dup, sort
    return sorted(set(x for x in vals if 1 <= x <= 200))


def inject_size_into_s3_key(s3_key: str, size_pct: int) -> str:
    """
    Insert '/size=<NN>' before the shard/sha1 tail. Works with your current layout:
      .../sizes=<menu_id>/<shard>/<sha1>/output_result.json.gz
    -> .../sizes=<menu_id>/size=<NN>/<shard>/<sha1>/output_result.json.gz
    """
    # find last two slashes before filename
    last = s3_key.rfind("/")
    if last < 0:
        return s3_key
    last2 = s3_key.rfind("/", 0, last)
    if last2 < 0:
        return s3_key
    base = s3_key[:last2]         # up to .../sizes=<menu_id>
    tail = s3_key[last2:]         # /<shard>/<sha1>/output_result.json.gz
    return f"{base}/size={int(size_pct)}{tail}"

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

    def to_msgs(row):
        """
        Expand ONE manifest row into MULTIPLE SQS entries, one per bet size.
        Falls back to a single entry with size=33 if bet_sizes is empty.
        """
        # 1) parse bet sizes
        sizes_pct = parse_bet_sizes_cell(row.get("bet_sizes"))
        if not sizes_pct:
            sizes_pct = [33]  # safe default; your worker can still legalize at runtime

        entries = []

        for size_pct in sizes_pct:
            # 2) build params subset
            params = {}
            for k in PARAM_WHITELIST:
                if k in row and pd.notna(row[k]):
                    v = _coerce_param(k, row[k])
                    if v is not None:
                        params[k] = v
            # stamp concrete size for this job
            params["size_pct"] = int(size_pct)

            # 3) size-aware s3_key
            s3_key = inject_size_into_s3_key(str(row["s3_key"]), int(size_pct))

            # 4) unique Id per (sha1,size)
            msg = {
                "sha1": str(row["sha1"]),
                "s3_key": s3_key,
                "params": params,
            }
            entries.append({
                "Id": f"{str(row['sha1'])[:72]}_{int(size_pct)}",
                "MessageBody": json.dumps(msg)
            })

        return entries

    if args.dry_run:
        print("🧪 dry-run: not sending to SQS")
        sample_rows = df.head(min(2, len(df)))
        for _, r in sample_rows.iterrows():
            em = to_msgs(r)
            print(f"row sha1={r['sha1']} → {len(em)} msgs")
            for e in em[:min(3, len(em))]:
                body = json.loads(e["MessageBody"])
                print(" ", e["Id"], body["s3_key"], "size_pct=", body["params"].get("size_pct"))
        return

    sqs = boto3.client(
        "sqs",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    n = len(df)
    sent = 0
    retries = 0
    entries_buf = []

    def _send_batch(entries):
        nonlocal sent, retries
        if not entries:
            return
        resp2 = {"Successful": []}
        try:
            resp = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=entries)
            failed = resp.get("Failed", []) or []
            succ = len(resp.get("Successful", []))

            if failed:
                fail_ids = {f["Id"] for f in failed}
                retry_entries = [e for e in entries if e["Id"] in fail_ids]
                if retry_entries:
                    time.sleep(0.5)
                    retries += 1
                    resp2 = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=retry_entries)
                    succ += len(resp2.get("Successful", []))
                    failed2 = resp2.get("Failed", []) or []
                    if failed2:
                        print(f"⚠️  retry still failed for {len(failed2)} entries "
                              f"(e.g. {failed2[0] if failed2 else ''})")

            sent += succ

        except botocore.exceptions.BotoCoreError as e:
            print(f"⚠️  SQS error: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")

    # Expand each manifest row into one-or-more messages and send in batches of ≤10
    for _, row in df.iterrows():
        entries_buf.extend(to_msgs(row))
        while len(entries_buf) >= args.batch:
            batch = entries_buf[:args.batch]
            entries_buf = entries_buf[args.batch:]
            _send_batch(batch)
            if sent and (sent % (args.batch * 20) == 0):
                print(f"… progress: sent≈{sent}")

    # Flush any remainder
    if entries_buf:
        _send_batch(entries_buf)

    print(f"✅ submitted ~{sent} messages to SQS  (retries: {retries})")

if __name__ == "__main__":
    main()
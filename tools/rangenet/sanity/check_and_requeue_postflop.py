#!/usr/bin/env python3
# tools/rangenet/sanity/check_and_requeue_postflop.py
import argparse, os, json, time, io, gzip, re
from pathlib import Path

import boto3
import botocore
import pandas as pd

# ---------- Fast checks ----------
SENTINEL = re.compile(r'"nodes"\s*:\s*\{\s*"root"\s*:|\"root\"\s*:|\"tree\"\s*:')
SMALL_BYTES = 2048  # flag tiny objects

PARAM_WHITELIST = [
    "street", "pot_bb", "effective_stack_bb", "positions", "bet_sizing_id",
    "range_ip", "range_oop",
    "accuracy", "max_iter", "allin_threshold",
    "node_key", "solver_version",
    "board", "board_cluster_id",
    "parent_sha1", "parent_node_key", "line_key",
    "street_root_state", "turn_card", "river_card",
]

def s3_head(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404 or e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise

def read_range(s3, bucket, key, max_bytes=131072):
    """Return (ok, is_gz, raw_bytes[:max_bytes])."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{max_bytes-1}")
        b = obj["Body"].read()
        return True, key.endswith(".gz"), b
    except botocore.exceptions.ClientError as e:
        return False, False, b""

def quick_json_sanity(raw: bytes, gz: bool) -> bool:
    try:
        data = gzip.decompress(raw) if gz and raw.startswith(b"\x1f\x8b") else raw
        text = data.decode("utf-8", errors="ignore")
        return bool(SENTINEL.search(text))
    except Exception:
        return False

def to_msg(row, solver_cfg):
    params = {}
    for k in PARAM_WHITELIST:
        if k in row and pd.notna(row[k]):
            v = row[k]
            if k in ("street","max_iter","board_cluster_id"):
                try: v = int(v)
                except: continue
            elif k in ("pot_bb","effective_stack_bb","accuracy","allin_threshold"):
                try: v = float(v)
                except: continue
            elif not isinstance(v, (int,float,str)):
                v = str(v)
            params[k] = v

    # overrides from config
    acc = solver_cfg.get("accuracy", params.get("accuracy"))
    mi  = solver_cfg.get("max_iter", solver_cfg.get("max_iterations", params.get("max_iter")))
    ath = solver_cfg.get("allin_threshold", params.get("allin_threshold"))
    if acc is not None: params["accuracy"] = float(acc)
    if mi  is not None: params["max_iter"] = int(mi)
    if ath is not None: params["allin_threshold"] = float(ath)
    if "version" in solver_cfg:
        params["solver_version"] = solver_cfg["version"]

    sha1 = str(row["sha1"])
    body = {
        "sha1": sha1,
        "s3_key": str(row["s3_key"]),
        "params": params,
    }
    return {"Id": sha1[:80], "MessageBody": json.dumps(body)}

def load_yaml_like(cfg_name_or_path: str):
    # Tiny helper: support either dotted model name (via your loader) or raw yaml path
    try:
        from ml.config import load_model_config
        return load_model_config(cfg_name_or_path)
    except Exception:
        import yaml
        with open(cfg_name_or_path, "r") as f:
            return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser("Reconcile postflop outputs (S3) against submit manifest and optionally requeue missing/bad.")
    ap.add_argument("--manifest", required=True, help="Submit manifest parquet used to queue jobs")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="", help="Common S3 prefix (optional; only for listing extras)")
    ap.add_argument("--outdir", default="data/artifacts/checks_postflop")
    ap.add_argument("--config", default="rangenet/postflop", help="YAML or model key with worker settings (for requeue)")
    ap.add_argument("--queue_url", default=None, help="SQS queue URL to requeue")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only process first N manifest rows")
    ap.add_argument("--requeue", action="store_true", help="actually requeue missing/bad rows")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.manifest)
    if args.limit is not None:
        df = df.head(int(args.limit))

    # sanity: required columns
    need = ["sha1","s3_key"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise SystemExit(f"Manifest missing columns: {miss}")

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION") or "eu-central-1")

    found_rows, missing_rows, tiny_rows, parse_bad_rows = [], [], [], []

    for i, r in df.iterrows():
        s3_key = str(r["s3_key"])
        # try exact key + common alternates
        candidates = [s3_key]
        if s3_key.endswith(".json.gz"):
            candidates.append(s3_key[:-3])  # .json
        elif s3_key.endswith(".json"):
            candidates.append(s3_key + ".gz")

        head = meta = None
        actual_key = None
        for k in candidates:
            head = s3_head(s3, args.bucket, k)
            if head:
                actual_key = k
                break

        if not head:
            missing_rows.append(r.to_dict() | {"expected_key": s3_key})
            continue

        size = int(head.get("ContentLength", 0))
        if size < SMALL_BYTES:
            tiny_rows.append(r.to_dict() | {"key": actual_key, "size": size})
            continue

        ok, gz, raw = read_range(s3, args.bucket, actual_key)
        if not ok or not quick_json_sanity(raw, gz):
            parse_bad_rows.append(r.to_dict() | {"key": actual_key, "size": size})
            continue

        found_rows.append(r.to_dict() | {"key": actual_key, "size": size})

    # Write reports
    def dump(name, rows):
        if rows:
            df2 = pd.DataFrame(rows)
            df2.to_csv(outdir / f"{name}.csv", index=False)

    dump("found", found_rows)
    dump("missing", missing_rows)
    dump("tiny", tiny_rows)
    dump("parse_bad", parse_bad_rows)

    tot = len(df)
    n_found, n_missing = len(found_rows), len(missing_rows)
    n_tiny, n_bad = len(tiny_rows), len(parse_bad_rows)

    print(f"\n=== Reconcile summary ===")
    print(f"Expected (manifest): {tot}")
    print(f"FOUND:   {n_found}")
    print(f"MISSING: {n_missing}")
    print(f"TINY:    {n_tiny}")
    print(f"PARSE_BAD: {n_bad}")
    print(f"CSV reports in: {outdir}\n")

    # Optional requeue
    if args.requeue and args.queue_url:
        cfg = load_yaml_like(args.config)
        solver_cfg = (cfg.get("worker") or {}) if isinstance(cfg, dict) else {}
        if args.batch < 1 or args.batch > 10:
            raise SystemExit("--batch must be 1..10 for SQS")

        sqs = boto3.client(
            "sqs",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )

        # candidates to requeue = missing + tiny + parse_bad
        to_requeue = []
        for rows in (missing_rows, tiny_rows, parse_bad_rows):
            for d in rows:
                # Rebuild a pandas-like row dict by merging back original row fields:
                sha1 = d.get("sha1")
                base = df.loc[df["sha1"] == sha1]
                if base.empty:
                    continue
                to_requeue.append(base.iloc[0])

        # de-dup by sha1
        if to_requeue:
            seen = set(); entries = []
            for r in to_requeue:
                sha = str(r["sha1"])
                if sha in seen: continue
                seen.add(sha)
                entries.append(to_msg(r, solver_cfg))

            # batch send
            sent = 0; retries = 0
            i = 0; n = len(entries)
            while i < n:
                chunk = entries[i:i+args.batch]
                resp2 = {"Successful": []}
                try:
                    resp = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=chunk)
                    failed = resp.get("Failed", []) or []
                    if failed:
                        ids = {f["Id"] for f in failed}
                        retry = [e for e in chunk if e["Id"] in ids]
                        if retry:
                            time.sleep(0.5)
                            retries += 1
                            resp2 = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=retry)
                    ok = len(resp.get("Successful", [])) + len(resp2.get("Successful", []))
                    sent += ok
                except Exception as e:
                    print(f"⚠️ requeue error @ {i}: {e}")
                i += args.batch

            print(f"🔁 Requeued {sent}/{len(entries)} items  (retries: {retries})")
        else:
            print("Nothing to requeue.")
    elif args.requeue and not args.queue_url:
        print("⚠️ --requeue requested but no --queue_url provided; skipped requeueing.")

if __name__ == "__main__":
    main()
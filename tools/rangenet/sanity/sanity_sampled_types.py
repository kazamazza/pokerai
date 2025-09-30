#!/usr/bin/env python3
import os, re, json, gzip, argparse, tempfile
import sys
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import (
    SolverPolicyParser, PolicyParseConfig, ACTION_VOCAB
)

# ---------- helpers ----------
KEY_RE = re.compile(
    r"^solver/outputs/(?P<ver>[^/]+)/"
    r"street=(?P<street>\d+)/pos=(?P<pos>[^/]+)/stack=(?P<stack>\d+)/pot=(?P<pot>\d+)/"
    r"board=(?P<board>[^/]+)/acc=(?P<acc>[^/]+)/sizes=(?P<sizes>[^/]+)/"
)

def parse_key(key: str):
    """Extract stack, pot, bet_sizing_id (sizes), and role suffix from an S3 key."""
    m = KEY_RE.match(key)
    if not m:
        return None
    d = m.groupdict()
    sizes = d["sizes"]            # e.g., 'srp_hu.PFR_IP', '3bet_hu.Aggressor_OOP', 'limped_multi.Any'
    stack = float(d["stack"])
    pot   = float(d["pot"])
    # role suffix comes after last dot if present; otherwise the entire sizes
    role  = sizes.split(".", 1)[1] if "." in sizes else sizes
    return {
        "stack_bb": stack,
        "pot_bb": pot,
        "bet_sizing_id": sizes,
        "role": role,
    }

def load_json_local(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path, "r") as f:
        return json.load(f)

def summarize_vec(vec):
    idx = {a:i for i,a in enumerate(ACTION_VOCAB)}
    out = []
    for a in ACTION_VOCAB:
        v = vec[idx[a]]
        if v > 0:
            out.append((a, v))
    # Sorted by mass desc for readability
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Sample one file per solve type and sanity-check with SolverPolicyParser")
    ap.add_argument("--bucket", default="pokeraistore")
    ap.add_argument("--prefix", default="solver/outputs/v1/")
    ap.add_argument("--max-keys", type=int, default=200000, help="Upper bound to scan")
    ap.add_argument("--region", default=os.getenv("AWS_REGION") or "eu-west-1")
    ap.add_argument("--profile", default=None, help="Optional AWS profile")
    ap.add_argument("--dry-run", action="store_true", help="List picks but do not download/parse")
    args = ap.parse_args()

    if args.profile:
        boto3.setup_default_session(profile_name=args.profile)

    s3 = boto3.client("s3", config=BotoConfig(region_name=args.region))

    # 1) List keys and group by bet_sizing_id (sizes)
    paginator = s3.get_paginator("list_objects_v2")
    picks = {}  # bet_sizing_id -> key
    scanned = 0

    try:
        for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not (key.endswith(".json") or key.endswith(".json.gz")):
                    continue
                meta = parse_key(key)
                if not meta:
                    continue
                bmid = meta["bet_sizing_id"]
                # take first occurrence
                if bmid not in picks:
                    picks[bmid] = key
                scanned += 1
                if scanned >= args.max_keys:
                    break
            if scanned >= args.max_keys:
                break
    except (BotoCoreError, ClientError) as e:
        print(f"ERROR listing S3: {e}")
        return

    if not picks:
        print("No matching keys found under the given prefix.")
        return

    print(f"Found {len(picks)} bet menu types (scanned ~{scanned} keys).")
    for bmid, key in sorted(picks.items()):
        print(f"  · {bmid:25s} → {key}")

    if args.dry_run:
        print("\nDRY-RUN: stopping before download/parse.")
        return

    # 2) Download + parse each pick
    print("\n=== Parsing one sample per bet menu type ===\n")
    parser = SolverPolicyParser()
    tmpdir = Path(tempfile.mkdtemp(prefix="sample_types_"))

    for bmid, key in sorted(picks.items()):
        meta = parse_key(key)
        if not meta:
            print(f"[skip] could not parse: {key}")
            continue

        role = meta["role"]          # e.g. PFR_IP, Aggressor_OOP, Any, Caller_OOP…
        pot  = meta["pot_bb"]
        stack= meta["stack_bb"]

        local = tmpdir / (Path(key).name)
        try:
            s3.download_file(args.bucket, key, str(local))
        except (BotoCoreError, ClientError) as e:
            print(f"[{bmid}] download ERROR: {e}")
            continue

        try:
            payload = load_json_local(local)
        except Exception as e:
            print(f"[{bmid}] JSON load ERROR: {e}")
            continue

        cfg = PolicyParseConfig(pot_bb=pot, stack_bb=stack, role=role)
        out = parser.parse(payload, cfg)

        print(f"[{bmid}] ok={out.ok} role={role} pot={pot} stack={stack}")
        if out.debug:
            print(f"  debug: {out.debug}")

        s = sum(out.vec)
        print(f"  sum: {s:.6f}")
        for a, v in summarize_vec(out.vec)[:8]:
            print(f"    {a:10s}: {v:.4f}")
        print()

if __name__ == "__main__":
    main()
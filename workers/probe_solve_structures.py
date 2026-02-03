from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import boto3


# ----------------------------
# S3 helpers
# ----------------------------

def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def list_keys(s3, bucket: str, prefix: str, limit: int) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            k = obj["Key"]
            if k.endswith(".json.gz"):
                keys.append(k)
                if limit and len(keys) >= limit:
                    return keys

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    return keys


def get_gz_json(s3, bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
        raw = gz.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


# ----------------------------
# Structure probing
# ----------------------------

def _sample_keys(d: Dict[str, Any], n: int = 20) -> List[str]:
    return list(d.keys())[:n]


def _find_first(d: Any, path: str = "", depth: int = 0, max_depth: int = 6) -> List[Tuple[str, Any]]:
    """
    Heuristic search for places strategy/node data might live.
    Returns list of (path, value) for interesting hits.
    """
    hits: List[Tuple[str, Any]] = []
    if depth > max_depth:
        return hits

    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k

            # keywords that often indicate structure
            if k.lower() in {"nodes", "tree", "strategy", "strategies", "root", "children", "lines", "actions"}:
                hits.append((p, v))

            hits.extend(_find_first(v, p, depth + 1, max_depth))
    elif isinstance(d, list):
        # only peek first few to avoid blowups
        for i, v in enumerate(d[:3]):
            p = f"{path}[{i}]"
            hits.extend(_find_first(v, p, depth + 1, max_depth))

    return hits


def _node_count_guess(obj: Dict[str, Any]) -> Optional[int]:
    """
    Try a few common formats.
    """
    # 1) obj["nodes"] is list/dict
    nodes = obj.get("nodes")
    if isinstance(nodes, list):
        return len(nodes)
    if isinstance(nodes, dict):
        return len(nodes)

    # 2) obj["tree"]["nodes"]
    tree = obj.get("tree")
    if isinstance(tree, dict):
        tn = tree.get("nodes")
        if isinstance(tn, list):
            return len(tn)
        if isinstance(tn, dict):
            return len(tn)

    return None


def _has_any_facing_hint(obj: Dict[str, Any]) -> bool:
    """
    Very rough heuristic: look for words or shapes that imply branching / responses.
    """
    txt = json.dumps(obj)[:200_000].lower()  # cap
    # "facing" might not exist; we look for evidence of child nodes / responses
    return any(w in txt for w in ["children", "child", "next", "branch", "node", "action_sequence", "line", "response"])


@dataclass
class ProbeResult:
    key: str
    top_keys: List[str]
    node_count: Optional[int]
    facing_hint: bool
    interesting_paths: List[str]


def probe_one(s3, bucket: str, key: str) -> ProbeResult:
    obj = get_gz_json(s3, bucket, key)

    top_keys = _sample_keys(obj, 30)
    node_count = _node_count_guess(obj)
    facing_hint = _has_any_facing_hint(obj)

    hits = _find_first(obj)
    # compress hits to readable lines
    interesting_paths: List[str] = []
    for p, v in hits[:25]:
        t = type(v).__name__
        if isinstance(v, (list, dict)):
            # add size hint
            try:
                sz = len(v)
                interesting_paths.append(f"{p} : {t}(len={sz})")
            except Exception:
                interesting_paths.append(f"{p} : {t}")
        else:
            s = str(v)
            interesting_paths.append(f"{p} : {t}={s[:80]}")

    return ProbeResult(
        key=key,
        top_keys=top_keys,
        node_count=node_count,
        facing_hint=facing_hint,
        interesting_paths=interesting_paths,
    )


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("Probe solver output json.gz structures in S3")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
    ap.add_argument("--prefix", default="solver/outputs/v1/")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--show-paths", action="store_true", help="Print interesting nested paths")

    args = ap.parse_args()

    s3 = s3_client(args.region)
    keys = list_keys(s3, args.bucket, args.prefix, args.limit)
    if not keys:
        print("No .json.gz keys found under prefix:", args.prefix)
        return

    print(f"Found {len(keys)} solve files (limited). Probing...\n")

    roots_only = 0
    likely_tree = 0

    for k in keys:
        r = probe_one(s3, args.bucket, k)

        print("=" * 90)
        print("KEY:", r.key)
        print("TOP:", r.top_keys)
        print("NODE_COUNT_GUESS:", r.node_count)
        print("FACING_HINT:", r.facing_hint)

        # classification heuristic
        if (r.node_count == 1) or (r.node_count is None and not r.facing_hint):
            roots_only += 1
            print("CLASS:", "LIKELY_ROOT_ONLY")
        else:
            likely_tree += 1
            print("CLASS:", "LIKELY_HAS_TREE/FACING")

        if args.show_paths:
            print("\nINTERESTING PATHS:")
            for p in r.interesting_paths:
                print("  -", p)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print(f"  total probed        : {len(keys)}")
    print(f"  likely root-only    : {roots_only}")
    print(f"  likely tree/facing  : {likely_tree}")
    print("=" * 90)


if __name__ == "__main__":
    main()
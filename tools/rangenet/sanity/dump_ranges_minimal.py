#!/usr/bin/env python3
# tools/rangenet/sanity/dump_ranges_minimal.py
"""
Stream-only minimal extractor:
- Reads output_result.json(.gz) directly from S3
- Extracts only the root node's ranges (IP/OOP)
- Converts to 169-vectors and writes a small parquet

Usage example:
  python tools/rangenet/sanity/dump_ranges_minimal.py \
    --bucket pokeraistore \
    --keys-file data/pilots_15.txt \
    --limit 15 \
    --workers 3 \
    --out data/artifacts/mini_ranges.parquet
"""

from __future__ import annotations
import argparse, boto3, io, gzip, json, os, re, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

# --------- small helpers (no numpy) ----------
_RANKS = "AKQJT98765432"

def hand_to_index_169(h: str) -> Optional[int]:
    if not h:
        return None
    h = h.strip()
    suited = None
    if len(h) == 2:
        r1, r2 = h[0], h[1]
    elif len(h) == 3:
        r1, r2 = h[0], h[1]; suited = h[2].lower()
    elif len(h) == 4:
        r1, s1, r2, s2 = h[0], h[1], h[2], h[3]; suited = "s" if s1 == s2 else "o"
    else:
        return None
    if r1 not in _RANKS or r2 not in _RANKS:
        return None
    i, j = _RANKS.index(r1), _RANKS.index(r2)
    if i == j:
        row, col = i, j
    else:
        if suited == "s":
            if i > j: i, j = j, i
            row, col = j, i
        elif suited == "o":
            if i < j: i, j = j, i
            row, col = i, j
        else:
            return None
    idx = row * 13 + col
    return idx if 0 <= idx < 169 else None

def vec169_from_map(rmap: Dict[str, float]) -> List[float]:
    v = [0.0] * 169
    for h, p in (rmap or {}).items():
        idx = hand_to_index_169(str(h))
        if idx is not None:
            try:
                x = float(p)
                if x < 0: x = 0.0
                if x > 1: x = 1.0
                v[idx] = x
            except Exception:
                pass
    return v

# --------- streaming JSON root extraction ----------
_TARGET_KEYS = [r'"nodes"\s*:\s*\{\s*"root"\s*:', r'"root"\s*:', r'"tree"\s*:']
_SENTINEL = re.compile("|".join(_TARGET_KEYS))

def _is_gzip_magic(b2: bytes) -> bool:
    return b2.startswith(b"\x1f\x8b")

def _iter_decompressed_chunks(body_stream, chunk_size=1<<16):
    # body_stream: botocore.response.StreamingBody or file-like
    # Peek first 2 bytes to detect gzip, then stream
    start = body_stream.read(2)
    rest = body_stream.read
    if not start:
        return
    # Rebuild a new stream with the peeked bytes
    prepend = io.BytesIO(start + rest(0))
    # For StreamingBody we can't "unread"; wrap a chained stream
    class _Chain(io.RawIOBase):
        def __init__(self, first: bytes, body):
            self._first = io.BytesIO(first)
            self._body = body
        def read(self, n=-1):
            b = self._first.read(n)
            if b:
                return b
            return self._body.read(n)
    chained = _Chain(start, body_stream)
    if _is_gzip_magic(start):
        raw = gzip.GzipFile(fileobj=io.BufferedReader(chained))
    else:
        raw = io.BufferedReader(chained)
    while True:
        chunk = raw.read(chunk_size)
        if not chunk:
            break
        yield chunk.decode("utf-8", errors="ignore")

def _extract_root_snippet_from_stream(body_stream) -> str:
    """
    Return a JSON snippet containing ONLY the root object:
      - nodes.root {...}   OR
      - root {...}         OR
      - tree {...}
    We find the first of those keys, then bracket-balance from the following '{'.
    """
    buf = ""
    start_idx = None
    depth = 0
    seen_key = False

    for txt in _iter_decompressed_chunks(body_stream):
        buf += txt
        if not seen_key:
            m = _SENTINEL.search(buf)
            if not m:
                # Trim to avoid unbounded growth while we search
                if len(buf) > 300_000:
                    buf = buf[-200_000:]
                continue
            i = m.end()
            while i < len(buf) and buf[i] != "{":
                i += 1
            if i >= len(buf):
                # need more data
                continue
            start_idx = i
            seen_key = True

        # balance from start_idx
        if start_idx is not None:
            depth = 0
            for j in range(start_idx, len(buf)):
                ch = buf[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return buf[start_idx:j+1]

        # keep buffer manageable
        if len(buf) > 2_000_000:
            # keep last 1MB to handle objects crossing chunk boundaries
            buf = buf[-1_000_000:]

    raise RuntimeError("Could not locate root object in solver JSON")

def _extract_ranges_from_root_obj(root_obj: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Normalize common shapes -> (ip_map, oop_map)."""
    def norm_map(x):
        out = {}
        if isinstance(x, dict):
            for k, v in x.items():
                try: out[str(k)] = float(v)
                except Exception: pass
        return out

    # nodes[root] shape with 'ranges'
    if isinstance(root_obj.get("ranges"), dict):
        ip = norm_map(root_obj["ranges"].get("ip"))
        oop = norm_map(root_obj["ranges"].get("oop"))
        if ip or oop:
            return ip, oop

    # actors.ip.range shape
    actors = root_obj.get("actors")
    if isinstance(actors, dict):
        ip = norm_map((actors.get("ip") or {}).get("range") or (actors.get("ip") or {}).get("ranges"))
        oop = norm_map((actors.get("oop") or {}).get("range") or (actors.get("oop") or {}).get("ranges"))
        if ip or oop:
            return ip, oop

    # flat fields (rare)
    ip = norm_map(root_obj.get("range_ip"))
    oop = norm_map(root_obj.get("range_oop"))
    return ip, oop

# --------- key parsing helpers ----------
def _parts_from_keypath(k: str) -> dict:
    # expects .../street=1/pos=BBvSB/stack=25/pot=2/board=8dQdQc/acc=0.01/sizes=.../<sha1>/output_result.json.gz
    parts = {}
    for seg in Path(k).parts:
        if "=" in seg:
            a, b = seg.split("=", 1)
            parts[a] = b
    return parts

def _pick_root(doc: dict) -> dict:
    # common shapes: nodes.root / root / tree
    if isinstance(doc.get("nodes"), dict) and isinstance(doc["nodes"].get("root"), dict):
        return doc["nodes"]["root"]
    for k in ("root", "tree"):
        if isinstance(doc.get(k), dict):
            return doc[k]
    # fallback: whole doc (won't find ranges but prevents crash)
    return doc

# --------- per-key worker ----------
def _process_key(bucket: str, key: str, region: str, parser: str = "json") -> Optional[dict]:
    import io, json, gzip, time, sys
    import boto3

    s3 = boto3.client("s3", region_name=region)
    t0 = time.time()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]

        # load JSON (gzip-aware)
        if key.endswith(".gz"):
            data = gzip.decompress(body.read())
            doc = json.loads(data)
        else:
            doc = json.loads(body.read().decode("utf-8"))

        # >>> FIX: zoom into the actual root node <<<
        root_obj = _pick_root(doc)

        # extract ranges from the root object
        ip_map, oop_map = _extract_ranges_from_root_obj(root_obj)
        r_ip  = vec169_from_map(ip_map)
        r_oop = vec169_from_map(oop_map)

        # (optional sanity) bail if both zero vectors
        if not any(r_ip) and not any(r_oop):
            sys.stderr.write(f"[warn] empty ranges in {key} (check bet/ctx or JSON shape)\n")

        meta = _parts_from_keypath(key)
        return {
            "street": int(meta.get("street", "1") or "1"),
            "positions": meta.get("pos", ""),
            "bet_sizing_id": meta.get("sizes", ""),
            "board": meta.get("board", ""),
            "effective_stack_bb": float(meta.get("stack", "0") or "0"),
            "pot_bb": float(meta.get("pot", "0") or "0"),
            "range_ip_169": r_ip,
            "range_oop_169": r_oop,
            "s3_key": key,
        }
    except Exception as e:
        sys.stderr.write(f"[skip] s3://{bucket}/{key}: {e}\n")
        return None

# --------- main ----------
def main():
    ap = argparse.ArgumentParser("Minimal fast extractor: stream root ranges → parquet")
    ap.add_argument("--bucket", required=True, help="S3 bucket")
    ap.add_argument("--prefix", default="solver/outputs/v1/", help="S3 prefix (used only if --keys-file missing)")
    ap.add_argument("--keys-file", help="Text file with one S3 key per line (preferred)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N keys (0 = no limit)")
    ap.add_argument("--workers", type=int, default=2, help="Threaded fetch/parse; keep small on laptops (1–4)")
    ap.add_argument("--region", default=os.getenv("AWS_REGION") or "eu-central-1")
    ap.add_argument("--out", default="data/artifacts/mini_ranges.parquet")
    args = ap.parse_args()

    # collect candidate keys
    keys: List[str] = []
    if args.keys_file:
        with open(args.keys_file, "r", encoding="utf-8") as f:
            for line in f:
                k = line.strip()
                if not k or k.startswith("#"):
                    continue
                keys.append(k)
    else:
        s3 = boto3.client("s3", region_name=args.region)
        sys.stderr.write(f"[info] listing s3://{args.bucket}/{args.prefix} …\n")
        resp = s3.list_objects_v2(Bucket=args.bucket, Prefix=args.prefix, MaxKeys=10_000)
        while True:
            for obj in resp.get("Contents", []):
                k = obj["Key"]
                if k.endswith("output_result.json") or k.endswith("output_result.json.gz"):
                    keys.append(k)
            if resp.get("IsTruncated"):
                resp = s3.list_objects_v2(
                    Bucket=args.bucket, Prefix=args.prefix,
                    ContinuationToken=resp["NextContinuationToken"], MaxKeys=10_000
                )
            else:
                break

    if args.limit and len(keys) > args.limit:
        keys = keys[:args.limit]

    if not keys:
        print("No keys to process.")
        return

    print(f"[info] parser=json(streamed), workers={args.workers}, total_keys={len(keys)}")
    rows: List[dict] = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_process_key, args.bucket, k, args.region) for k in keys]
        for fut in as_completed(futs):
            row = fut.result()
            done += 1
            if row:
                rows.append(row)
            if done % 1 == 0:
                elapsed = int((time.time() - t0) * 1000)
                print(f"[progress] {done}/{len(keys)}  (rows={len(rows)})   elapsed={elapsed}ms")

    if not rows:
        print("No rows extracted.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"✅ wrote {len(df)} rows → {out}")
    try:
        print(df[["positions","bet_sizing_id","board"]].head(10).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
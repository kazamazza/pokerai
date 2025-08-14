# tools/preflop/check_pair_coverage.py
import boto3, os, re
from collections import defaultdict

BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
PREFIX = "preflop/ranges/profile=GTO/exploit=GTO/multiway=HU/pop=REGULAR/"
s3 = boto3.client("s3")

def list_keys(prefix):
    keys, token = [], None
    while True:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, ContinuationToken=token) if token else \
               s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        for o in resp.get("Contents", []):
            keys.append(o["Key"])
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return keys

def parse(fname):
    m = re.search(r"([A-Z]+)_vs_([A-Z]+)_(\d+)bb\.json\.gz$", fname)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None

def main():
    open_keys = list_keys(PREFIX + "action=OPEN/")
    vso_keys  = list_keys(PREFIX + "action=VS_OPEN/")

    open_set = set(k for k in (parse(k) for k in open_keys) if k)
    vso_set  = set((b,a,s) for (a,b,s) in (parse(k) for k in vso_keys) if (a,b,s))

    missing_pairs = []
    for a,b,s in open_set:
        if (a,b,s) and ((a,b,s) not in open_set or (a,b,s) not in open_set):
            pass  # just to mirror logic
        if (a,b,s) not in open_set or (b,a,s) not in vso_set:
            missing_pairs.append((a,b,s))

    print(f"Pairs missing VS_OPEN (or OPEN) files: {len(missing_pairs)}")
    for a,b,s in missing_pairs[:20]:
        print(f"  need OPEN {a}_vs_{b}_{s}bb & VS_OPEN {b}_vs_{a}_{s}bb")

if __name__ == "__main__":
    main()
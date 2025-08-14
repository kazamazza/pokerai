import os, io, gzip, json, re, itertools
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError

# ====== CONFIG ======
REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")

PROFILE = "GTO"
EXPLOIT = "GTO"
MULTIWAY = "HU"
POP = "REGULAR"

STACKS = [10, 20, 40, 75, 100, 150, 200]
POSITIONS = ["UTG","MP","CO","BTN","SB","BB"]
MATCHUPS = [
    ("MP","UTG"),
    ("CO","UTG"), ("CO","MP"),
    ("BTN","UTG"), ("BTN","MP"), ("BTN","CO"),
    ("SB","UTG"), ("SB","MP"), ("SB","CO"), ("SB","BTN"),
    ("BB","UTG"), ("BB","MP"), ("BB","CO"), ("BB","BTN"),
    ("BB","SB"),
    ("SB","BB"),
]

# Allowed action buckets per context
ALLOWED = {
    "OPEN":     {"open","fold","limp"},  # add/remove limp if you disallow it
    "VS_OPEN":  {"fold","call","3bet","4bet","jam","defend","overcall"},
}

# Canonical 169 tokens (pairs, suited, offsuit)
RANKS = "AKQJT98765432"
CANON_169 = set([r+r for r in RANKS] +
                [a+b+"s" for a in RANKS for b in RANKS if RANKS.index(a)<RANKS.index(b)] +
                [a+b+"o" for a in RANKS for b in RANKS if RANKS.index(a)<RANKS.index(b)])

# ====== S3 ======
s3 = boto3.client("s3", region_name=REGION)

def preflop_prefix(action):
    return (f"preflop/ranges/"
            f"profile={PROFILE}/exploit={EXPLOIT}/multiway={MULTIWAY}/pop={POP}/action={action}/")

def build_key(action, ip, oop, stack):
    fname = f"{ip}_vs_{oop}_{stack}bb.json.gz"
    return preflop_prefix(action) + fname

def load_json_gz(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    by = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        return json.loads(gz.read().decode("utf-8"))

def list_keys(prefix):
    keys=[]
    token=None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix)
        if token: kw["ContinuationToken"]=token
        resp = s3.list_objects_v2(**kw)
        for c in resp.get("Contents", []):
            keys.append(c["Key"])
        if not resp.get("IsTruncated"): break
        token = resp["NextContinuationToken"]
    return keys

def is_169_token(x: str) -> bool:
    # Simple sanity: AA, AKs, KQo, etc (already normalized)
    return x in CANON_169

def validate_file_schema(doc, action_ctx, key, issues):
    if not isinstance(doc, dict) or "actions" not in doc:
        issues.append({"key":key,"err":"missing_actions_dict"})
        return {}
    actions = doc.get("actions") or {}
    if not isinstance(actions, dict):
        issues.append({"key":key,"err":"actions_not_dict"})
        return {}
    # Check allowed buckets
    for a in actions.keys():
        if a.lower() not in ALLOWED[action_ctx]:
            issues.append({"key":key,"err":"illegal_bucket", "bucket":a})
    # Hand tokens
    seen=set()
    for a, hands in actions.items():
        if not isinstance(hands, list):
            issues.append({"key":key,"err":"bucket_not_list","bucket":a})
            continue
        for h in hands:
            if not isinstance(h, str):
                issues.append({"key":key,"err":"hand_not_str","hand":repr(h)})
                continue
            if not is_169_token(h):
                issues.append({"key":key,"err":"bad_hand_token","hand":h,"bucket":a})
            if h in seen:
                issues.append({"key":key,"err":"duplicate_hand","hand":h})
            seen.add(h)
    return actions

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def run_audit():
    issues=[]
    stats=defaultdict(int)

    # quick existence map for VS_OPEN
    vs_open_keys = set(list_keys(preflop_prefix("VS_OPEN")))

    for ip, oop in MATCHUPS:
        for stack in STACKS:
            open_key   = build_key("OPEN",   ip,  oop, stack)
            defend_key = build_key("VS_OPEN", oop, ip, stack)  # note reversed roles

            # 1) Both files should exist
            for k in [open_key, defend_key]:
                try:
                    s3.head_object(Bucket=BUCKET, Key=k)
                except ClientError as e:
                    issues.append({"key":k,"err":"missing_file"})
                    continue

            # 2) Load & validate schema
            try:
                open_doc = load_json_gz(open_key)
            except Exception as e:
                issues.append({"key":open_key,"err":"open_load_fail","detail":str(e)})
                continue
            try:
                vso_doc = load_json_gz(defend_key)
            except Exception as e:
                issues.append({"key":defend_key,"err":"vso_load_fail","detail":str(e)})
                continue

            open_actions = validate_file_schema(open_doc, "OPEN", open_key, issues)
            vso_actions  = validate_file_schema(vso_doc,  "VS_OPEN", defend_key, issues)

            if not open_actions or not vso_actions:
                continue

            # 3) Extract sets
            ip_open = set(open_actions.get("open", []))
            oop_defend = set(vso_actions.get("call", [])) \
                       | set(vso_actions.get("3bet", [])) \
                       | set(vso_actions.get("4bet", [])) \
                       | set(vso_actions.get("jam", [])) \
                       | set(vso_actions.get("defend", [])) \
                       | set(vso_actions.get("overcall", []))

            # 4) Sanity metrics
            stats["pairs_checked"] += 1
            if len(ip_open) == 0:
                issues.append({"key":open_key,"err":"empty_open"})
            if len(oop_defend) == 0:
                issues.append({"key":defend_key,"err":"empty_defend"})

            # 5) Jaccard should generally be < 1
            jac = jaccard(ip_open, oop_defend)
            if jac >= 0.98 and len(ip_open) > 0:  # suspiciously identical
                issues.append({"pair":f"{ip}_vs_{oop}_{stack}bb","err":"identical_ranges","jaccard":jac})

            # 6) Basic positional monotonicity (optional heuristic)
            # e.g., UTG open should not be larger than BTN open by a big margin—skip here unless you want thresholds.

    # Report
    if not issues:
        print("✅ All checks passed.")
    else:
        print(f"⚠️ Found {len(issues)} anomalies. First 20:")
        for item in issues[:20]:
            print(" -", item)

    # Optionally dump a JSONL for full review
    out = "preflop_audit_issues.jsonl"
    with open(out, "w") as f:
        for it in issues:
            f.write(json.dumps(it) + "\n")
    print(f"📝 Wrote detailed issues to {out}")

if __name__ == "__main__":
    run_audit()
#!/usr/bin/env python
import gzip, json, boto3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import SolverPolicyParser, PolicyParseConfig, ACTION_VOCAB

# Replace with your file
S3_KEY = "solver/outputs/v1/street=1/pos=COvSB/stack=150/pot=20/board=QcJdQd/acc=0.03/sizes=3bet_hu.Aggressor_OOP/a1/a13815724a3812981301644c8ec6ae6cc17841f6/output_result.json.gz"
LOCAL = Path("debug_live_oop.json.gz")

# Download once
s3 = boto3.client("s3")
bucket = "pokeraistore"
s3.download_file(bucket, S3_KEY, str(LOCAL))

# Load JSON (compressed)
with gzip.open(LOCAL, "rt") as f:
    payload = json.load(f)

cfg = PolicyParseConfig(
    pot_bb=20.0,        # from the path
    stack_bb=150.0,     # from the path
    role="Aggressor_OOP"  # from bet_sizing_id
)

out = SolverPolicyParser().parse(payload, cfg)

print("ok:", out.ok)
print("debug:", out.debug)
print("sum=", sum(out.vec))
for a, i in enumerate(ACTION_VOCAB):
    if out.vec[a] > 0:
        print(f"{i:10s}: {out.vec[a]:.4f}")
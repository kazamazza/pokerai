
import json, gzip, io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import PolicyParseConfig, SolverPolicyParser
from ml.models.policy_consts import ACTION_VOCAB

# --- deps you already have in this repo ---


# --- config inferred from the S3 path you gave ---
S3_URI   = "s3://pokeraistore/solver/outputs/v1/street=1/pos=UTGvBB/stack=25/pot=8/board=8c3sTc/acc=0.02/sizes=srp_hu.PFR_IP/5b/5b628ea7bfb6c66d8fb6d6d477b05ce7f1956a38/output_result.json.gz"
POT_BB   = 8.0
STACK_BB = 25.0
ROLE     = "PFR_IP"   # aggressor at root

# --- load JSON from S3 (works for .gz or plain json) ---
def load_json_from_s3(uri: str):
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(uri, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":  # gz
        return json.loads(gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8"))
    return json.loads(raw.decode("utf-8"))

payload = load_json_from_s3(S3_URI)

# --- parse policy ---
cfg  = PolicyParseConfig(pot_bb=POT_BB, stack_bb=STACK_BB, role=ROLE)
out  = SolverPolicyParser().parse(payload, cfg)

idx = {a:i for i,a in enumerate(ACTION_VOCAB)}
v   = out.vec
raise_mass = sum(v[idx[a]] for a in ("RAISE_150","RAISE_200","RAISE_300","ALLIN") if a in idx)

print("ok:", out.ok)
print("mode:", out.debug.get("mode"))
print("sum=", round(sum(v), 6))
print("CALL:", round(v[idx.get("CALL", 0)], 6))
print("FOLD:", round(v[idx.get("FOLD", 0)], 6))
for a in ("BET_25","BET_33","BET_50","BET_66","BET_75","BET_100"):
    if a in idx: print(f"{a}: {round(v[idx[a]], 6)}")
for a in ("RAISE_150","RAISE_200","RAISE_300","ALLIN"):
    if a in idx: print(f"{a}: {round(v[idx[a]], 6)}")
print("RAISE_TOTAL:", round(raise_mass, 6))
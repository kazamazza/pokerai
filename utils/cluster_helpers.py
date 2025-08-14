import gzip
import io
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import boto3
from boto3 import s3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

FLOP_CLUSTER_MAP_PATH = Path("data/flop/flop_cluster_map.json")
BUCKET = os.getenv("AWS_BUCKET_NAME", "PokerAIStore")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def _load_flop_cluster_map() -> Dict[str, int]:
    with FLOP_CLUSTER_MAP_PATH.open("r") as f:
        return json.load(f)

def _valid_clusters_and_representatives(cluster_map: Dict[str, int]) -> Tuple[set, Dict[int, str]]:
    valid_clusters = set(cluster_map.values())
    first_board_for_cluster: Dict[int, str] = {}
    for board, cid in cluster_map.items():
        if cid not in first_board_for_cluster:
            first_board_for_cluster[cid] = board
    return valid_clusters, first_board_for_cluster

def _board_for_cluster_id(cid: int, rep_map: Dict[int, str]) -> str:
    try:
        return rep_map[cid]
    except KeyError:
        raise ValueError(f"No representative board found for cluster_id={cid}")


# =========================
# S3 preflop loader (gz)
# =========================

def build_preflop_filename(ip: str, oop: str, stack_bb: int) -> str:
    # Canonical naming: IP_vs_OOP_<stack>bb.json.gz
    return f"{ip}_vs_{oop}_{stack_bb}bb.json.gz"

def build_preflop_s3_key(
    ip: str,
    oop: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str,
) -> str:
    filename = build_preflop_filename(ip, oop, stack_bb)
    return (
        "preflop/ranges/"
        f"profile={villain_profile}/exploit={exploit_setting}/multiway={multiway_context}/"
        f"pop={population_type}/action={action_context}/{filename}"
    )

def load_preflop_json_from_s3(
    ip: str,
    oop: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str,
) -> Dict:
    key = build_preflop_s3_key(
        ip, oop, stack_bb, villain_profile, exploit_setting, multiway_context, population_type, action_context
    )
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
    except ClientError as e:
        raise FileNotFoundError(f"Could not fetch preflop file: s3://{BUCKET}/{key} ({e})")

    by = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        text = gz.read().decode("utf-8")
    return json.loads(text)

# --- VS_OPEN key candidates & loader ---

def _vs_open_key_candidates(ip_defender: str, oop_opener: str, stack_bb: int,
                            profile: str, exploit: str, multiway: str, pop: str) -> list[str]:
    base = (f"preflop/ranges/profile={profile}/exploit={exploit}/multiway={multiway}/"
            f"pop={pop}/action=VS_OPEN/")
    # Your bucket currently uses the same ordering as OPEN: IP_vs_OOP
    k1 = f"{ip_defender}_vs_{oop_opener}_{stack_bb}bb.json.gz"  # defender_vs_opener  (matches your listing)
    # Fallback in case some sets were produced the other way:
    k2 = f"{oop_opener}_vs_{ip_defender}_{stack_bb}bb.json.gz"  # opener_vs_defender
    return [base + k1, base + k2]

def _load_vs_open_doc_any(ip_defender: str, oop_opener: str, stack_bb: int,
                          profile: str, exploit: str, multiway: str, pop: str) -> tuple[dict, str]:
    last_err = None
    for key in _vs_open_key_candidates(ip_defender, oop_opener, stack_bb, profile, exploit, multiway, pop):
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            by = obj["Body"].read()
            with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
                return json.loads(gz.read().decode("utf-8")), key
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"VS_OPEN not found for defender={ip_defender}, opener={oop_opener}, stack={stack_bb}bb. "
        f"Tried both orderings. Last error: {last_err}"
    )
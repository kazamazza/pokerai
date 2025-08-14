import gzip, io, json
import os
from typing import Tuple, List, Dict, Set

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from utils.builders import build_preflop_s3_key_components

# Buckets to collect from preflop JSON
OPEN_OPENER_ACTIONS: Set[str] = {"open"}
VS_OPEN_DEFENDER_ACTIONS: Set[str] = {"call", "defend", "3bet", "4bet", "jam", "overcall"}

VS_3BET_OPENER_CONTINUE: Set[str] = {"call"}          # opener continues vs 3bet
VS_4BET_DEFENDER_CONTINUE: Set[str] = {"call"}        # 3bettor continues vs 4bet
VS_OPEN_3BET_RANGE: Set[str] = {"3bet"}              # 3bettor range comes from VS_OPEN
VS_3BET_4BET_RANGE: Set[str] = {"4bet"}

load_dotenv()
REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")

s3 = boto3.client("s3", region_name=REGION)

def _load_preflop_json_from_s3(*, ip: str, oop: str, stack_bb: int,
                               profile: str, exploit: str, multiway: str, pop: str, action: str) -> Dict:
    key = build_preflop_s3_key_components(
        ip=ip, oop=oop, stack_bb=stack_bb,
        villain_profile=profile, exploit_setting=exploit,
        multiway_context=multiway, population_type=pop, action_context=action
    )
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
    except ClientError as e:
        raise FileNotFoundError(f"Missing preflop file: s3://{BUCKET}/{key} ({e})")

    by = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        text = gz.read().decode("utf-8")
    return json.loads(text)

def _collect(doc: Dict, wanted: Set[str]) -> List[str]:
    acts = doc.get("actions") or {}
    out: List[str] = []
    seen = set()
    for k, v in acts.items():
        if k.lower() in wanted:
            for c in v:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out

def extract_ip_oop_ranges_open(*, ip: str, oop: str, stack_bb: int,
                               villain_profile: str, exploit_setting: str,
                               multiway_context: str, population_type: str) -> Tuple[List[str], List[str]]:
    opener = _load_preflop_json_from_s3(
        ip=ip, oop=oop, stack_bb=stack_bb, profile=villain_profile, exploit=exploit_setting,
        multiway=multiway_context, pop=population_type, action="OPEN"
    )
    defender = _load_preflop_json_from_s3(
        ip=oop, oop=ip, stack_bb=stack_bb, profile=villain_profile, exploit=exploit_setting,
        multiway=multiway_context, pop=population_type, action="VS_OPEN"
    )
    ip_range  = _collect(opener, OPEN_OPENER_ACTIONS)
    oop_range = _collect(defender, VS_OPEN_DEFENDER_ACTIONS)
    return ip_range, oop_range

def extract_ip_oop_ranges_vs_open(*, ip: str, oop: str, stack_bb: int,
                                  villain_profile: str, exploit_setting: str,
                                  multiway_context: str, population_type: str) -> Tuple[List[str], List[str]]:
    # same deps as OPEN; the tuple is the same SRP pairing
    return extract_ip_oop_ranges_open(
        ip=ip, oop=oop, stack_bb=stack_bb,
        villain_profile=villain_profile, exploit_setting=exploit_setting,
        multiway_context=multiway_context, population_type=population_type
    )

def extract_ip_oop_ranges_vs_3bet(
    *, ip: str, oop: str, stack_bb: int,
    villain_profile: str, exploit_setting: str,
    multiway_context: str, population_type: str
) -> Tuple[List[str], List[str]]:
    """
    3-bet pot created when defender (OOP) 3bets opener (IP).
    We need:
      - 3bettor range: from VS_OPEN (defender vs opener) -> '3bet'
      - caller range:  from VS_3BET (opener vs defender)  -> 'call'
    Final ranges are **flop participants after preflop is done**:
      IP  = opener's *calling* range vs 3bet
      OOP = defender's *3bet* range
    """
    # defender's VS_OPEN (oop vs ip): get 3bet range
    vso_def = _load_preflop_json_from_s3(
        ip=oop, oop=ip, stack_bb=stack_bb,
        profile=villain_profile, exploit=exploit_setting,
        multiway=multiway_context, pop=population_type,
        action="VS_OPEN"
    )
    oop_3bet_range = _collect(vso_def, VS_OPEN_3BET_RANGE)

    # opener's VS_3BET (ip vs oop): get call range
    v3b_opener = _load_preflop_json_from_s3(
        ip=ip, oop=oop, stack_bb=stack_bb,
        profile=villain_profile, exploit=exploit_setting,
        multiway=multiway_context, pop=population_type,
        action="VS_3BET"
    )
    ip_call_range = _collect(v3b_opener, VS_3BET_OPENER_CONTINUE)

    return ip_call_range, oop_3bet_range


def extract_ip_oop_ranges_vs_4bet(
    *, ip: str, oop: str, stack_bb: int,
    villain_profile: str, exploit_setting: str,
    multiway_context: str, population_type: str
) -> Tuple[List[str], List[str]]:
    """
    4-bet pot created when opener (IP) 4bets vs defender’s (OOP) 3bet.
    We need:
      - 4bettor range: from VS_3BET (opener vs defender) -> '4bet'
      - caller range:  from VS_4BET (defender vs opener) -> 'call'
    Final flop participants:
      IP  = opener's *4bet* range
      OOP = defender's *call* range vs 4bet
    """
    # opener's VS_3BET (ip vs oop): get 4bet range
    v3b_opener = _load_preflop_json_from_s3(
        ip=ip, oop=oop, stack_bb=stack_bb,
        profile=villain_profile, exploit=exploit_setting,
        multiway=multiway_context, pop=population_type,
        action="VS_3BET"
    )
    ip_4bet_range = _collect(v3b_opener, VS_3BET_4BET_RANGE)

    # defender's VS_4BET (oop vs ip): get call range
    v4b_def = _load_preflop_json_from_s3(
        ip=oop, oop=ip, stack_bb=stack_bb,
        profile=villain_profile, exploit=exploit_setting,
        multiway=multiway_context, pop=population_type,
        action="VS_4BET"
    )
    oop_call_vs4b_range = _collect(v4b_def, VS_4BET_DEFENDER_CONTINUE)

    return ip_4bet_range, oop_call_vs4b_range


# 🔁 UPDATE the EXTRACTORS dict to include the new contexts
EXTRACTORS = {
    "OPEN":     extract_ip_oop_ranges_open,
    "VS_OPEN":  extract_ip_oop_ranges_vs_open,
    "VS_3BET":  extract_ip_oop_ranges_vs_3bet,
    "VS_4BET":  extract_ip_oop_ranges_vs_4bet,
}
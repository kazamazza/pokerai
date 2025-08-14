import gzip
import io
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from infra.storage.s3_uploader import S3Uploader
from utils.builders import build_preflop_s3_key
from utils.combos import get_169_combo_list
from utils.equity import expand_combo_string, compute_hand_vs_range_equity
from utils.range_classifier import classify_by_equity
from utils.range_extraction import _is_valid_open_pair
from utils.range_utils import get_preflop_range

load_dotenv()
s3 = S3Uploader()


def generate_single_range(config: dict) -> tuple[str, Path]:
    """
    Build preflop payload and return (s3_key, gzipped_bytes). No S3 or filesystem.
    """
    ip_position      = config["ip_position"]
    oop_position     = config["oop_position"]
    stack_bb         = config["stack_bb"]
    villain_profile  = config["villain_profile"]
    exploit_setting  = config["exploit_setting"]
    multiway_context = config["multiway_context"]
    population_type  = config["population_type"]
    action_context   = config["action_context"]

    combos = get_169_combo_list()

    opponent_hands = get_preflop_range(
        position=oop_position,
        stack_depth=stack_bb,
        villain_profile=villain_profile,
        exploit_setting=exploit_setting,
        multiway_context=multiway_context,
        population_type=population_type,
        action_context=action_context
    )

    if not opponent_hands or not all(len(c) == 4 for c in opponent_hands):
        raise ValueError(f"Invalid range for {oop_position} @ {stack_bb}bb")

    if action_context == "OPEN" and not _is_valid_open_pair(ip_position, oop_position):
        raise ValueError(f"Invalid OPEN pairing: {ip_position} vs {oop_position}")

    buckets = defaultdict(list)
    for hand in combos:
        hero_combos = expand_combo_string(hand)
        if not hero_combos:
            continue

        equity = compute_hand_vs_range_equity(hand, opponent_hands)
        action = classify_by_equity(
            equity=equity,
            position=ip_position,
            stack_bb=stack_bb,
            villain_profile=villain_profile,
            exploit_setting=exploit_setting,
            multiway_context=multiway_context,
            population_type=population_type,
            action_context=action_context
        )
        buckets[action].append(hand)

    payload = {"meta": config, "actions": buckets}

    s3_key = build_preflop_s3_key(payload["meta"])
    unique = f"{uuid.uuid4().hex}_{os.getpid()}"
    temp_path = Path("/tmp") / f"{unique}.json.gz"

    with gzip.open(temp_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return s3_key, temp_path


if __name__ == "__main__":
    raise RuntimeError("Run this from sqs_worker.py with input config")

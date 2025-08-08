import gzip
import json
import os
import uuid
from pathlib import Path
from collections import defaultdict
from typing import Dict
from dotenv import load_dotenv

from infra.storage.s3_uploader import S3Uploader
from utils.combos import get_169_combo_list
from utils.equity import expand_combo_string, compute_hand_vs_range_equity
from utils.range_classifier import classify_by_equity
from utils.range_utils import get_preflop_range

load_dotenv()
s3 = S3Uploader()


def generate_single_range(config: Dict):
    """
    Generate a single preflop range and upload the result to S3.
    """
    ip_position = config["ip_position"]
    oop_position = config["oop_position"]
    stack_bb = config["stack_bb"]
    villain_profile = config["villain_profile"]
    exploit_setting = config["exploit_setting"]
    multiway_context = config["multiway_context"]
    population_type = config["population_type"]
    action_context = config["action_context"]

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

    # Format JSON output
    payload = {
        "meta": config,
        "actions": buckets
    }

    filename = f"{ip_position}_vs_{oop_position}_{stack_bb}bb.json.gz"
    s3_key = f"preflop/ranges/profile={villain_profile}/exploit={exploit_setting}/multiway={multiway_context}/pop={population_type}/action={action_context}/{filename}"

    # make temp unique per worker/process to avoid clobber
    unique = f"{uuid.uuid4().hex}_{os.getpid()}"
    temp_path = Path("/tmp") / f"{unique}_{filename}"

    with gzip.open(temp_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    s3.upload_file(temp_path, s3_key)  # your wrapper takes Path; ok
    temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise RuntimeError("Run this from sqs_worker.py with input config")

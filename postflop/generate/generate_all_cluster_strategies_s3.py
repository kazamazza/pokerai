from __future__ import annotations
import gzip
import json
import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.range_extraction import extract_ip_oop_ranges_for_open
from features.types import STACK_BUCKETS
from preflop.matchups import MATCHUPS
from utils.build_cluster_strategy_object import build_cluster_strategy_object
from utils.cluster_helpers import _load_flop_cluster_map, _valid_clusters_and_representatives, _board_for_cluster_id

load_dotenv()

REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")

s3 = boto3.client("s3", region_name=REGION)

# Output directory for temporary local gz files before upload (kept on disk briefly)
OUTPUT_DIR = Path("postflop/strategy_templates")

def generate_all_cluster_strategies():
    cluster_map = _load_flop_cluster_map()
    valid_clusters, rep_board = _valid_clusters_and_representatives(cluster_map)

    # Fixed axes for now (per your earlier plan). Expand later if desired.
    villain_profile = "GTO"
    exploit_setting = "GTO"
    multiway_context = "HU"
    population_type = "REGULAR"
    action_context = "OPEN"

    total_written = 0

    for cluster_id in sorted(valid_clusters):
        board = _board_for_cluster_id(cluster_id, rep_board)

        for ip_position, oop_position in MATCHUPS:
            for stack_bb in STACK_BUCKETS:

                print(
                    f"\n🧠 Cluster {cluster_id} ({board}) | {ip_position} vs {oop_position} @ {stack_bb}bb"
                    f"\n    → {villain_profile}/{exploit_setting}/{multiway_context}/{population_type}/{action_context}"
                )

                try:
                    ip_range, oop_range = extract_ip_oop_ranges_for_open(
                        ip=ip_position,
                        oop=oop_position,
                        stack_bb=stack_bb,
                        villain_profile=villain_profile,
                        exploit_setting=exploit_setting,
                        multiway_context=multiway_context,
                        population_type=population_type,
                    )
                except Exception as e:
                    print(f"[ERROR] Range extraction failed: {e}")
                    continue

                if not ip_range or not oop_range:
                    print(f"[SKIP] Empty ranges (IP={len(ip_range)}, OOP={len(oop_range)}).")
                    continue

                strategy = build_cluster_strategy_object(
                    cluster_id=cluster_id,
                    board=board,
                    ip_range=ip_range,
                    oop_range=oop_range,
                )

                out_dir = (
                    OUTPUT_DIR / villain_profile / exploit_setting / multiway_context / population_type / action_context
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                file_out = f"{ip_position}_vs_{oop_position}_{stack_bb}bb_cluster_{cluster_id}.json.gz"
                local_out_path = out_dir / file_out

                try:
                    with gzip.open(local_out_path, "wt", encoding="utf-8") as f:
                        json.dump(strategy.model_dump(), f, indent=2)
                    print(f"✅ Saved (gz): {local_out_path}")
                except Exception as e:
                    print(f"[ERROR] Local gzip write failed: {e}")
                    continue

                s3_out_key = (
                    "postflop/strategy_templates/"
                    f"profile={villain_profile}/exploit={exploit_setting}/multiway={multiway_context}/"
                    f"pop={population_type}/action={action_context}/{file_out}"
                )
                try:
                    s3.upload_file(str(local_out_path), BUCKET, s3_out_key)
                    print(f"🚀 Uploaded: s3://{BUCKET}/{s3_out_key}")
                    total_written += 1
                except ClientError as e:
                    print(f"[ERROR] Upload failed: {e}")

    print(f"\n🎯 Done. Total strategies written: {total_written}")


# === Entry Point ===
if __name__ == "__main__":
    generate_all_cluster_strategies()
import os
import json
import gzip
import sys
import time
import traceback
import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from utils.pot_calculators import compute_srp_flop_pot_bb, compute_3bet_flop_pot_bb, compute_4bet_flop_pot_bb, \
    get_flop_pot_from_context

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from preflop.generate_ranges import generate_single_range
from simulation.solver_interface import run_solver
from utils.builders import build_postflop_solved_s3_key, UPLOAD_CLUSTER_TEMPLATES, build_cluster_template_s3_key, \
    build_cluster_strategy_object, SOLVE_FROM_BOTH_SIDES, build_solver_request_from_cluster, \
    build_preflop_s3_key_components
from utils.cluster_helpers import _board_for_cluster_id, _load_flop_cluster_map, _valid_clusters_and_representatives
from utils.range_extraction import extract_ip_oop_ranges_for_open
from workers.base import SQSWorker

# === ENV / AWS ===
load_dotenv()
REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")

s3 = boto3.client("s3", region_name=REGION)

# === Fixed axes for cluster phase (as agreed to keep search space sane) ===
CLUSTER_VILLAIN_PROFILE = "GTO"
CLUSTER_EXPLOIT_SETTING = "GTO"
CLUSTER_MULTIWAY_CONTEXT = "HU"
CLUSTER_POPULATION_TYPE = "REGULAR"
CLUSTER_ACTION_CONTEXT = "OPEN"

# === Where cluster templates are written locally before upload ===
OUTPUT_DIR = Path("postflop/strategy_templates")

# ---------- S3 helpers ----------
def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


def generate_clusters_for_matchup_stack(ip_position: str, oop_position: str, stack_bb: int) -> int:
    """
    For one (IP,OOP,stack), generate all cluster templates (optional upload),
    run solver (both perspectives), and upload the solved artifact for each cluster.
    Returns number of solved artifacts uploaded.
    """
    cluster_map = _load_flop_cluster_map()
    valid_clusters, rep_board = _valid_clusters_and_representatives(cluster_map)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = 0

    for cluster_id in sorted(valid_clusters):
        board = _board_for_cluster_id(cluster_id, rep_board)

        print(
            f"\n🧠 Cluster {cluster_id} ({board}) | {ip_position} vs {oop_position} @ {stack_bb}bb"
            f"\n    → {CLUSTER_VILLAIN_PROFILE}/{CLUSTER_EXPLOIT_SETTING}/"
            f"{CLUSTER_MULTIWAY_CONTEXT}/{CLUSTER_POPULATION_TYPE}/{CLUSTER_ACTION_CONTEXT}"
        )

        # -------- Ranges
        try:
            ip_range, oop_range = extract_ip_oop_ranges_for_open(
                ip=ip_position,
                oop=oop_position,
                stack_bb=stack_bb,
                villain_profile=CLUSTER_VILLAIN_PROFILE,
                exploit_setting=CLUSTER_EXPLOIT_SETTING,
                multiway_context=CLUSTER_MULTIWAY_CONTEXT,
                population_type=CLUSTER_POPULATION_TYPE,
            )
        except Exception as e:
            print(f"[ERROR] Range extraction failed: {e}")
            continue

        if not ip_range or not oop_range:
            print(f"[SKIP] Empty ranges (IP={len(ip_range)}, OOP={len(oop_range)}).")
            continue

        # -------- Template object (schema-compliant)
        strategy = build_cluster_strategy_object(
            cluster_id=cluster_id,
            board=board,
            ip_range=ip_range,
            oop_range=oop_range,
        )

        # -------- (Optional) upload template
        if UPLOAD_CLUSTER_TEMPLATES:
            try:
                tmpl_key = build_cluster_template_s3_key(ip=ip_position, oop=oop_position, stack_bb=stack_bb, cluster_id=cluster_id)
                if not s3_exists(tmpl_key):
                    local_tmpl = Path("/tmp") / f"cluster_tmpl_{uuid.uuid4().hex}.json.gz"
                    with gzip.open(local_tmpl, "wt", encoding="utf-8") as f:
                        json.dump(strategy.model_dump(), f, indent=2)
                    s3.upload_file(str(local_tmpl), BUCKET, tmpl_key)
                    print(f"🗂️  Uploaded template: s3://{BUCKET}/{tmpl_key}")
                    local_tmpl.unlink(missing_ok=True)
            except Exception as e:
                print(f"[WARN] Template upload skipped: {e}")

        # -------- Solver enrich (lightweight: opponent range vectors)
        solver_aux = {}
        try:
            OPEN_SIZE = 2.5
            THREEBET_SZ = 8.5
            FOURBET_SZ = 22.0
            TABLE_PLAYERS = 6
            ANTE_BB = 0.0

            # You fixed these for clustering; if later you vary them per cfg, pass cfg["action_context"] instead
            pot_guess_bb = get_flop_pot_from_context(
                action_context=CLUSTER_ACTION_CONTEXT,  # e.g. "OPEN"
                opener_pos=ip_position,  # opener is IP in your OPEN clusters
                caller_pos=oop_position,  # defender is OOP
                open_size_bb=OPEN_SIZE,
                threebet_size_bb=THREEBET_SZ,
                fourbet_size_bb=FOURBET_SZ,
                num_players=TABLE_PLAYERS,
                ante_bb=ANTE_BB
            )

            if SOLVE_FROM_BOTH_SIDES:
                req_ip  = build_solver_request_from_cluster(strategy, stack_bb=stack_bb, pot_size_bb=pot_guess_bb, hero_role="IP")
                vec_ip, _  = run_solver(req_ip)

                req_oop = build_solver_request_from_cluster(strategy, stack_bb=stack_bb, pot_size_bb=pot_guess_bb, hero_role="OOP")
                vec_oop, _ = run_solver(req_oop)

                solver_aux = {
                    "opp_vec_ip":  vec_ip,   # 169 floats
                    "opp_vec_oop": vec_oop,  # 169 floats
                }
            else:
                req_ip  = build_solver_request_from_cluster(strategy, stack_bb=stack_bb, pot_size_bb=pot_guess_bb, hero_role="IP")
                vec_ip, _ = run_solver(req_ip)
                solver_aux = {"opp_vec_ip": vec_ip}

        except Exception as e:
            print(f"[SOLVER] skipped due to error: {e}")
            solver_aux = {"solver_error": str(e)}

        # -------- Final postflop solved artifact
        payload = {
            "meta": {
                "cluster_id": cluster_id,
                "board": board,
                "ip_position": ip_position,
                "oop_position": oop_position,
                "stack_bb": stack_bb,
                "villain_profile": CLUSTER_VILLAIN_PROFILE,
                "exploit_setting": CLUSTER_EXPLOIT_SETTING,
                "multiway_context": CLUSTER_MULTIWAY_CONTEXT,
                "population_type": CLUSTER_POPULATION_TYPE,
                "action_context": CLUSTER_ACTION_CONTEXT,
            },
            "ranges": {
                "ip":  strategy.ip_range,
                "oop": strategy.oop_range,
            },
            "solver": solver_aux,
        }

        out_key = build_postflop_solved_s3_key(ip=ip_position, oop=oop_position, stack_bb=stack_bb, cluster_id=cluster_id)
        try:
            if not s3_exists(out_key):
                tmp_out = Path("/tmp") / f"postflop_{uuid.uuid4().hex}.json.gz"
                with gzip.open(tmp_out, "wt", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                s3.upload_file(str(tmp_out), BUCKET, out_key)
                tmp_out.unlink(missing_ok=True)
                uploaded += 1
                print(f"🚀 Uploaded solved: s3://{BUCKET}/{out_key}")
            else:
                print(f"🟢 Exists: s3://{BUCKET}/{out_key}")
        except Exception as e:
            print(f"[ERROR] Upload solved failed: {e}")

    return uploaded


def maybe_run_clusters_for_this_preflop(cfg: dict) -> None:
    """
    After writing a preflop file, check if BOTH required preflop files exist for this matchup/stack:
      - OPEN:    ip_vs_oop
      - VS_OPEN: oop_vs_ip
    If both exist, generate all clusters for (ip, oop, stack).
    """
    ip = cfg["ip_position"]
    oop = cfg["oop_position"]
    stack = cfg["stack_bb"]

    # Keys we require to run clusters (fixed axes for clustering stage)
    open_key = build_preflop_s3_key_components(
        ip=ip, oop=oop, stack_bb=stack,
        villain_profile=CLUSTER_VILLAIN_PROFILE,
        exploit_setting=CLUSTER_EXPLOIT_SETTING,
        multiway_context=CLUSTER_MULTIWAY_CONTEXT,
        population_type=CLUSTER_POPULATION_TYPE,
        action_context="OPEN",
    )

    vs_open_key = build_preflop_s3_key_components(
        ip=oop, oop=ip, stack_bb=stack,  # note the reversal here
        villain_profile=CLUSTER_VILLAIN_PROFILE,
        exploit_setting=CLUSTER_EXPLOIT_SETTING,
        multiway_context=CLUSTER_MULTIWAY_CONTEXT,
        population_type=CLUSTER_POPULATION_TYPE,
        action_context="VS_OPEN"
    )

    print(f"🔎 Checking deps:\n  OPEN    → s3://{BUCKET}/{open_key}\n  VS_OPEN → s3://{BUCKET}/{vs_open_key}")
    if s3_exists(open_key) and s3_exists(vs_open_key):
        print("✅ Dependencies satisfied → generating clusters now…")
        count = generate_clusters_for_matchup_stack(ip, oop, stack)
        print(f"🎯 Clusters generated/uploaded: {count}")
    else:
        print("⏳ Dependencies not yet satisfied — will run when the counterpart file lands.")


# ---------- Preflop SQS handler (pipeline style) ----------
def handle_preflop_task(message_body: str) -> bool:
    """
    1) Build & upload the preflop JSON (gz).
    2) If its counterpart file exists (OPEN vs VS_OPEN pair), immediately generate clusters for this matchup/stack.
    """
    try:
        cfg = json.loads(message_body)

        # Step 1: produce preflop file (your existing function)
        s3_key, temp_path = generate_single_range(cfg)

        # Idempotency guard
        try:
            head = s3.head_object(Bucket=BUCKET, Key=s3_key)
            if head.get("ContentLength", 0) > 0:
                print(f"🟢 Already exists: s3://{BUCKET}/{s3_key} — skipping upload")
                Path(temp_path).unlink(missing_ok=True)
                # Even if it exists, still attempt to chain cluster step
                maybe_run_clusters_for_this_preflop(cfg)
                return True
        except ClientError:
            pass

        # Upload
        s3.upload_file(str(temp_path), BUCKET, s3_key)

        # Verify
        for _ in range(5):
            try:
                head = s3.head_object(Bucket=BUCKET, Key=s3_key)
                if head.get("ContentLength", 0) > 0:
                    print(f"✅ Verified: s3://{BUCKET}/{s3_key}")
                    Path(temp_path).unlink(missing_ok=True)
                    # Step 2: attempt cluster step now that our file is present
                    maybe_run_clusters_for_this_preflop(cfg)
                    return True
            except Exception as e:
                print(f"⚠️ head_object retry: {e}")
            time.sleep(0.4)

        print(f"❌ Verification failed: s3://{BUCKET}/{s3_key}")
        return False

    except Exception as e:
        print(f"❌ Task failed: {e}")
        traceback.print_exc()
        return False


# ---------- Entry ----------
if __name__ == "__main__":
    # Use your existing SQSWorker wrapper
    worker = SQSWorker(
        handler=handle_preflop_task,
        max_threads=1,   # keep small to avoid solver contention downstream when you add postflop
        batch_size=1
    )
    worker.run()
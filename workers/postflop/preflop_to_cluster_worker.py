import os
import json
import gzip
import sys
import time
import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from postflop.extractors import EXTRACTORS
from postflop.flop_clusters import load_flop_cluster_map, valid_clusters_and_reps, board_for_cluster_id
from utils.cluster_plan import iter_cluster_axes
from postflop.deps import DEPS
from utils.pot_calculators import get_flop_pot_from_context
from preflop.generate_ranges import generate_single_range
from simulation.solver_interface import run_solver
from utils.builders import build_postflop_solved_s3_key, UPLOAD_CLUSTER_TEMPLATES, build_cluster_template_s3_key, \
    build_cluster_strategy_object, build_solver_request_from_cluster
from workers.base import SQSWorker

load_dotenv()
REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
QUEUE_URL = os.getenv("PRE_FLOP_QUEUE_URL")

s3 = boto3.client("s3", region_name=REGION)

# === Where cluster templates are written locally before upload ===
OUTPUT_DIR = Path("postflop/strategy_templates")

# ---- tiny util
def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False

def generate_clusters_for_axes(*, context: str, ip: str, oop: str, stack_bb: int,
                               prof: str, expl: str, mw: str, pop: str) -> int:
    cluster_map = load_flop_cluster_map()
    valid, reps = valid_clusters_and_reps(cluster_map)

    extractor = EXTRACTORS.get(context)
    if not extractor:
        print(f"[SKIP] No extractor for context={context}")
        return 0

    pot_guess_bb = get_flop_pot_from_context(context)
    uploaded = 0

    for cid in sorted(valid):
        board = board_for_cluster_id(cid, reps)
        print(f"\n🧠 Cluster {cid} ({board}) | {context} | {ip} vs {oop} @ {stack_bb}bb"
              f"\n    → {prof}/{expl}/{mw}/{pop}")

        # ranges
        try:
            ip_range, oop_range = extractor(
                ip=ip, oop=oop, stack_bb=stack_bb,
                villain_profile=prof, exploit_setting=expl,
                multiway_context=mw, population_type=pop
            )
        except Exception as e:
            print(f"[ERROR] Range extraction failed: {e}")
            continue

        if not ip_range or not oop_range:
            print(f"[SKIP] Empty ranges (IP={len(ip_range)}, OOP={len(oop_range)}).")
            continue

        strategy = build_cluster_strategy_object(cluster_id=cid, board=board, ip_range=ip_range, oop_range=oop_range)

        # optional template upload
        if UPLOAD_CLUSTER_TEMPLATES:
            try:
                tmpl_key = build_cluster_template_s3_key(ip=ip, oop=oop, stack_bb=stack_bb, cluster_id=cid,
                                                         context=context, profile=prof, exploit=expl, multiway=mw, pop=pop)
                if not s3_exists(tmpl_key):
                    p = Path("/tmp") / f"cluster_tmpl_{uuid.uuid4().hex}.json.gz"
                    with gzip.open(p, "wt", encoding="utf-8") as f: f.write(strategy.model_dump_json(indent=2))
                    s3.upload_file(str(p), BUCKET, tmpl_key); p.unlink(missing_ok=True)
                    print(f"🗂️  Uploaded template: s3://{BUCKET}/{tmpl_key}")
            except Exception as e:
                print(f"[WARN] Template upload skipped: {e}")

        # solver (lightweight)
        solver_aux = {}
        try:
            req_ip  = build_solver_request_from_cluster(strategy=strategy, stack_bb=stack_bb, pot_size_bb=pot_guess_bb, hero_role="IP")
            vec_ip, _  = run_solver(req_ip)
            req_oop = build_solver_request_from_cluster(strategy=strategy, stack_bb=stack_bb, pot_size_bb=pot_guess_bb, hero_role="OOP")
            vec_oop, _ = run_solver(req_oop)
            solver_aux = {"opp_vec_ip": vec_ip, "opp_vec_oop": vec_oop}
        except Exception as e:
            print(f"[SOLVER] skipped due to error: {e}")
            solver_aux = {"solver_error": str(e)}

        out_key = build_postflop_solved_s3_key(context=context, ip=ip, oop=oop, stack_bb=stack_bb,
                                               cluster_id=cid, profile=prof, exploit=expl, multiway=mw, pop=pop)
        try:
            if not s3_exists(out_key):
                p = Path("/tmp") / f"postflop_{uuid.uuid4().hex}.json.gz"
                payload = {
                    "meta": {"context": context, "cluster_id": cid, "board": board,
                             "ip_position": ip, "oop_position": oop, "stack_bb": stack_bb,
                             "villain_profile": prof, "exploit_setting": expl, "multiway_context": mw, "population_type": pop},
                    "ranges": {"ip": strategy.ip_range, "oop": strategy.oop_range},
                    "solver": solver_aux,
                }
                with gzip.open(p, "wt", encoding="utf-8") as f: json.dump(payload, f, indent=2)
                s3.upload_file(str(p), BUCKET, out_key); p.unlink(missing_ok=True)
                uploaded += 1
                print(f"🚀 Uploaded solved: s3://{BUCKET}/{out_key}")
            else:
                print(f"🟢 Exists: s3://{BUCKET}/{out_key}")
        except Exception as e:
            print(f"[ERROR] Upload solved failed: {e}")

    return uploaded

def _deps_satisfied(context: str, ip: str, oop: str, stack: int, prof: str, expl: str, mw: str, pop: str) -> bool:
    keys_fn = DEPS.get(context)
    if not keys_fn: return False
    for key in keys_fn(ip, oop, stack, prof, expl, mw, pop):
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
        except Exception:
            return False
    return True

def cluster_sweep() -> int:
    total = 0
    for (ctx, ip, oop, sb, prof, expl, mw, pop) in iter_cluster_axes():
        if not _deps_satisfied(ctx, ip, oop, sb, prof, expl, mw, pop):
            continue
        # probe: if any cluster 0 solved exists, assume tuple done (idempotent shortcut)
        probe = build_postflop_solved_s3_key(context=ctx, ip=ip, oop=oop, stack_bb=sb,
                                             cluster_id=0, profile=prof, exploit=expl, multiway=mw, pop=pop)
        if s3_exists(probe):
            continue
        total += generate_clusters_for_axes(context=ctx, ip=ip, oop=oop, stack_bb=sb,
                                            prof=prof, expl=expl, mw=mw, pop=pop)
    print(f"ℹ️ cluster_sweep uploaded {total} artifacts." if total else "ℹ️ cluster_sweep: nothing to do.")
    return total

# ---------- Preflop SQS handler (pipeline style)
def handle_preflop_task(message_body: str) -> bool:
    try:
        cfg = json.loads(message_body)
        s3_key, temp_path = generate_single_range(cfg)

        # If already exists:
        try:
            head = s3.head_object(Bucket=BUCKET, Key=s3_key)
            if head.get("ContentLength", 0) > 0:
                Path(temp_path).unlink(missing_ok=True)
                cluster_sweep()
                return True
        except ClientError:
            pass

        # Upload new
        s3.upload_file(str(temp_path), BUCKET, s3_key)

        # Verify + sweep
        for _ in range(5):
            try:
                head = s3.head_object(Bucket=BUCKET, Key=s3_key)
                if head.get("ContentLength", 0) > 0:
                    Path(temp_path).unlink(missing_ok=True)
                    cluster_sweep()
                    return True
            except Exception:
                time.sleep(0.4)

        print(f"❌ Verification failed: s3://{BUCKET}/{s3_key}")
        return False

    except Exception as e:
        print(f"❌ Task failed: {e}")
        import traceback; traceback.print_exc()
        return False

# ---------- Entry
if __name__ == "__main__":
    worker = SQSWorker(
        handler=handle_preflop_task,
        region=REGION,
        queue_url="https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-queue",
        dlq_url="https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-dlq",
        max_threads=1,
        batch_size=1,
    )
    worker.run()
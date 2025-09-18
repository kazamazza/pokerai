#!/usr/bin/env python3
import os, json, time, argparse, sys
import boto3, botocore
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

#!/usr/bin/env python3
import os, json, argparse, time
from typing import Tuple
import boto3
import pandas as pd

# ---- config loader (your project util) ----
from ml.utils.config import load_model_config

# ---------- extraction helpers ----------
def _role_from_menu_id(menu_id: str) -> str:
    m = (menu_id or "").strip()
    if "." in m:
        _, role = m.split(".", 1)
        return role
    return ""

def _infer_actor(row: pd.Series) -> str:
    vpos = str(row.get("villain_pos", "")).upper()
    if vpos in ("IP", "OOP"):
        return "ip" if vpos == "IP" else "oop"
    actor = str(row.get("actor", "")).lower()
    if actor in ("ip", "oop"):
        return actor
    pos = str(row.get("positions", "")).upper()
    if pos.startswith("IPV"): return "ip"
    if pos.startswith("OOPV"): return "oop"
    return "ip"

def _donk_available(menu_id: str, actor: str) -> bool:
    role = _role_from_menu_id(menu_id)
    grp  = (menu_id or "").split(".", 1)[0]
    if actor.lower() != "oop":
        return False
    if grp.startswith("limped_multi"): return False
    if "Caller_OOP" in role: return True
    if grp.startswith("limped_single"): return True
    return False

def _resolve_extraction(row: pd.Series) -> Tuple[str, str, str]:
    node_key = str(row.get("node_key") or "root")
    actor = _infer_actor(row)
    menu  = str(row.get("bet_sizing_id") or "")
    action_prefix = "DONK" if _donk_available(menu, actor) else "BET"
    return actor, action_prefix, node_key

# ---------- SQS envelope ----------
ENVELOPE_KEYS = ["sha1", "s3_key"]
PARAM_WHITELIST = [
    "street", "pot_bb", "effective_stack_bb", "positions", "bet_sizing_id",
    "range_ip", "range_oop",
    "accuracy", "max_iter", "allin_threshold",
    "node_key", "solver_version",
    "board", "board_cluster_id",
    "parent_sha1", "parent_node_key", "line_key",
    "street_root_state", "turn_card", "river_card",
]

def _row_to_msg(row: pd.Series, solver_cfg: dict):
    params = {}
    for k in PARAM_WHITELIST:
        if k in row and pd.notna(row[k]):
            v = row[k]
            if k in ("street", "max_iter", "board_cluster_id"):
                try: v = int(v)
                except: continue
            elif k in ("pot_bb", "effective_stack_bb", "accuracy", "allin_threshold"):
                try: v = float(v)
                except: continue
            elif not isinstance(v, (int, float, str)):
                v = str(v)
            params[k] = v

    # enforce config overrides
    acc = solver_cfg.get("accuracy", params.get("accuracy"))
    mi  = solver_cfg.get("max_iter", solver_cfg.get("max_iterations", params.get("max_iter")))
    ath = solver_cfg.get("allin_threshold", params.get("allin_threshold"))
    if acc is not None: params["accuracy"] = float(acc)
    if mi  is not None: params["max_iter"] = int(mi)
    if ath is not None: params["allin_threshold"] = float(ath)
    if "version" in solver_cfg:
        params["solver_version"] = solver_cfg["version"]

    msg = {
        "sha1": str(row["sha1"]),
        "s3_key": str(row["s3_key"]),
        "params": params,
    }
    return {"Id": str(row["sha1"])[:80], "MessageBody": json.dumps(msg)}

# ---------- selection logic ----------
def pick_unique_extraction_jobs(df: pd.DataFrame, per_key: int = 1, include_cluster: bool = False) -> pd.DataFrame:
    # ensure minimal columns exist
    need = ["bet_sizing_id", "positions"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Manifest missing required columns for selection: {missing}")

    if "street" not in df.columns: df = df.assign(street=1)
    if "node_key" not in df.columns: df = df.assign(node_key="root")

    # compute extraction tuple like the builder
    extra = df.apply(lambda r: _resolve_extraction(r), axis=1, result_type="expand")
    extra.columns = ["_actor", "_action_prefix", "_node_key"]
    dfx = pd.concat([df, extra], axis=1)

    key_cols = ["bet_sizing_id", "positions", "street", "_actor", "_action_prefix"]
    if include_cluster and "board_cluster_id" in dfx.columns:
        key_cols.append("board_cluster_id")

    # deterministic tie-breaker: prefer smallest stack, pot, then earliest board/sha
    sort_cols = [c for c in ("effective_stack_bb", "pot_bb", "board_cluster_id", "board", "sha1") if c in dfx.columns]
    if sort_cols:
        dfx = dfx.sort_values(sort_cols, kind="stable")

    pilots = (
        dfx.groupby(key_cols, dropna=False, sort=False)
           .head(int(per_key))
           .reset_index(drop=True)
    )
    return pilots

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Submit one job per extraction context to SQS")
    ap.add_argument("--manifest", default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--queue_url", required=True)
    ap.add_argument("--per_key", type=int, default=1)
    ap.add_argument("--include-cluster", action="store_true")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="rangenet/postflop")
    args = ap.parse_args()

    if args.batch < 1 or args.batch > 10:
        raise SystemExit("--batch must be between 1 and 10 (SQS limit)")

    cfg = load_model_config(args.config)
    solver_cfg = cfg.get("worker", {}) or {}

    df = pd.read_parquet(args.manifest)

    # sanity: manifest must have sha1 & s3_key (envelope used by worker)
    missing_env = [k for k in ENVELOPE_KEYS if k not in df.columns]
    if missing_env:
        raise SystemExit(f"Manifest missing required columns: {missing_env}")

    # select unique contexts
    pilots = pick_unique_extraction_jobs(df, per_key=args.per_key, include_cluster=args.include_cluster)

    print(f"🧪 selected {len(pilots)} pilot jobs (unique by context; per_key={args.per_key}; include_cluster={args.include_cluster})")

    if args.dry_run:
        show_cols = ["bet_sizing_id", "positions", "street", "_actor", "_action_prefix",
                     "effective_stack_bb", "pot_bb", "board_cluster_id", "board", "sha1", "s3_key"]
        show_cols = [c for c in show_cols if c in pilots.columns]
        print(pilots[show_cols].to_string(index=False, max_rows=60))
        return

    # SQS client
    sqs = boto3.client(
        "sqs",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION") or "eu-central-1",
    )

    n, i, sent, retries = len(pilots), 0, 0, 0
    while i < n:
        chunk = pilots.iloc[i:i + args.batch]
        entries = [_row_to_msg(r, solver_cfg) for _, r in chunk.iterrows()]

        # retry once on partial failure
        resp2 = {"Successful": []}
        try:
            resp = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=entries)
            failed = resp.get("Failed", []) or []
            if failed:
                fail_ids = {f["Id"] for f in failed}
                retry_entries = [e for e in entries if e["Id"] in fail_ids]
                if retry_entries:
                    time.sleep(0.5)
                    retries += 1
                    resp2 = sqs.send_message_batch(QueueUrl=args.queue_url, Entries=retry_entries)
        except Exception as e:
            print(f"⚠️  error submitting batch @ {i}: {e}")
            failed = []
        ok_count = len(resp.get("Successful", [])) + len(resp2.get("Successful", [])) if 'resp' in locals() else 0
        sent += ok_count
        i += args.batch

    print(f"✅ submitted ~{sent}/{n} pilot jobs (retries: {retries})")

if __name__ == "__main__":
    main()
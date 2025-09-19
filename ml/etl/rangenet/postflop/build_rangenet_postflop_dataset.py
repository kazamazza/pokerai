from __future__ import annotations
import json, gzip, io, os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import boto3

from ml.range.solvers.utils.solver_extract import extract_action_vector_169, pick_root, range_to_vec169
from ml.range.solvers.utils.solver_json_extract import hand_to_index_169


def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _s3_client(cfg: Mapping[str, Any]):
    region = _get(cfg, "aws.region") or os.getenv("AWS_REGION") or "eu-central-1"
    return boto3.client("s3", region_name=region)

def _local_cache_path(cfg: Mapping[str, Any], s3_key: str) -> Path:
    root = Path(_get(cfg, "worker.local_cache_dir", "data/solver_cache"))
    return (root / s3_key).resolve()

def _read_json_bytes(b: bytes, key_hint: str) -> dict:
    if key_hint.endswith(".gz"):
        b = gzip.GzipFile(fileobj=io.BytesIO(b)).read()
    return json.loads(b)

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3_key: str) -> dict:
    p = _local_cache_path(cfg, s3_key)
    if p.is_file():
        return _read_json_bytes(p.read_bytes(), p.name)
    bucket = _get(cfg, "aws.bucket") or os.getenv("AWS_BUCKET_NAME")
    s3 = _s3_client(cfg)
    body = s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)  # keep gz if gz
    return _read_json_bytes(body, s3_key)


# --- actor/action decision aligned with your bet-menus ---
def _role_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[1] if "." in m else ""

def _group_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[0] if "." in m else m

def _donk_available(menu_id: str, actor: str) -> bool:
    grp = _group_from_menu(menu_id)
    role = _role_from_menu(menu_id)
    if grp.startswith("limped_multi"):  # explicit no-donk case in your menu
        return False
    if (actor or "").lower() != "oop":
        return False
    return ("Caller_OOP" in role) or grp.startswith("limped_single")

def _infer_actor(row: pd.Series) -> str:
    # prefer explicit columns if present, else derive from positions
    actor = str(row.get("actor", "")).lower()
    if actor in ("ip","oop"):
        return actor
    role = _role_from_menu(str(row.get("bet_sizing_id","")))
    if role.endswith("_IP"):  return "ip"
    if role.endswith("_OOP"): return "oop"
    return "ip"

def _resolve_actor_action(row: pd.Series) -> Tuple[str, str]:
    actor = _infer_actor(row)
    menu_id = str(row.get("bet_sizing_id", ""))
    action = "DONK" if _donk_available(menu_id, actor) else "BET"
    return actor, action

def _split_positions(positions: str) -> Tuple[str, str]:
    # "BTNvBB" -> ("BTN","BB")
    s = str(positions).upper()
    if "V" in s:
        a, b = s.split("V", 1)
        return a, b
    return ("IP","OOP")

def _parse_child_label(label: str) -> tuple[str, Optional[int], str]:
    """
    Normalize a root child label into (action, bet_size_pct, raw).
    Examples:
      "CHECK"           -> ("CHECK", None, "CHECK")
      "BET 33"          -> ("BET", 33, "BET 33")
      "BET 66.0"        -> ("BET", 66, "BET 66.0")
      "DONK 33"         -> ("DONK", 33, "DONK 33")
      "RAISE 250"       -> ("RAISE", 250, "RAISE 250")
      "ALLIN"           -> ("ALLIN", None, "ALLIN")
    """
    raw = str(label).strip()
    up  = raw.upper()
    toks = up.split()
    if not toks:
        return ("UNKNOWN", None, raw)
    act = toks[0]
    size = None
    if len(toks) >= 2:
        try:
            size = int(round(float(toks[1])))
        except Exception:
            size = None
    return (act, size, raw)

def _extract_child_vec_169(root: dict, child_index: int) -> Optional[np.ndarray]:
    """
    Build a 169-vector with mass equal to the probability of choosing the *exact* child_index at root.
    """
    strat_map = (root.get("strategy") or {}).get("strategy") or {}
    if not strat_map:
        return None
    v = np.zeros(169, dtype=np.float32)
    any_set = False
    for hand, probs in strat_map.items():
        idx = hand_to_index_169(str(hand))
        if idx is None:
            continue
        try:
            p = float(probs[child_index])
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
            v[idx] = p
            any_set = True
        except Exception:
            pass
    return v if any_set and np.any(v) else None

def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Mapping[str, Any],
) -> None:
    """
    Read manifest → load each JSON → for every root child emit one training row:
      X: stack_bb, pot_bb, hero_pos, ip_pos, oop_pos, street, ctx, board/cluster,
         bet_sizing_id, actor, action, bet_size_pct, node_key
      Y: y_0..y_168 (policy mass for that action)
      W: weight
    """
    df = pd.read_parquet(manifest_path)

    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))
    normalize_y = bool(_get(cfg, "dataset.normalize_labels", True))

    # Required input columns
    need = ["s3_key","positions","street","effective_stack_bb","pot_bb","bet_sizing_id","ctx"]
    need += ["board_cluster_id"] if use_clusters else ["board"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    rows: List[Dict[str, Any]] = []
    skipped_empty = 0
    skipped_no_children = 0

    for _, r in df.iterrows():
        s3_key  = str(r["s3_key"])
        js      = _load_solver_json_local_or_s3(cfg, s3_key)
        node_key = str(r.get("node_key") or "root")

        root = pick_root(js, node_key=node_key)
        childrens = root.get("childrens") or {}
        child_labels = list(childrens.keys())
        if not child_labels:
            # Fallback: if there are no children but a root range exists,
            # you could still emit a single RANGE row — we skip by default.
            skipped_no_children += 1
            continue

        # Scenario fields
        ip_pos, oop_pos = _split_positions(str(r["positions"]))
        stack_bb  = int(round(float(r["effective_stack_bb"])))
        pot_bb    = float(r["pot_bb"])
        street    = int(r["street"])
        ctx       = str(r["ctx"])
        board_key = (int(r["board_cluster_id"]) if use_clusters else str(r["board"]))
        menu_id   = str(r["bet_sizing_id"])

        # Actor at root derived from menu (kept simple and consistent with earlier code)
        role = _role_from_menu(menu_id).upper()
        actor = ("ip" if role.endswith("_IP") else
                 "oop" if role.endswith("_OOP") else "ip")
        hero_pos = ip_pos if actor == "ip" else oop_pos

        # For each child, build a row
        for ci, lab in enumerate(child_labels):
            action, size_pct, raw_label = _parse_child_label(lab)
            vec = _extract_child_vec_169(root, ci)

            # Optional fallback to raw actor range if that child is missing mass
            if (vec is None) or (not np.any(vec)):
                continue  # per-action dataset: skip empty actions

            if normalize_y:
                s = float(vec.sum())
                if s > 0:
                    vec = vec / s

            row = {
                "stack_bb": stack_bb,
                "pot_bb": pot_bb,
                "hero_pos": hero_pos,
                "ip_pos": ip_pos,
                "oop_pos": oop_pos,
                "street": street,
                "ctx": ctx,
                "bet_sizing_id": menu_id,
                "actor": actor,              # who is acting at root
                "action": action,            # CHECK / BET / DONK / RAISE / ALLIN / ...
                "bet_size_pct": size_pct,    # integer percent if present (e.g., 33, 66, 250), else None
                "node_key": node_key,
                "weight": 1.0,
            }
            if use_clusters:
                row["board_cluster"] = int(board_key)
                row["board_cluster_id"] = int(board_key)
            else:
                row["board"] = str(board_key)

            # y_0..y_168
            for i, v in enumerate(vec.tolist()):
                row[f"y_{i}"] = float(v)

            rows.append(row)

    out = pd.DataFrame(rows)

    # Basic hygiene
    if len(out):
        # remove any NaNs in labels (shouldn’t happen, but safety)
        y_cols = [c for c in out.columns if c.startswith("y_")]
        out[y_cols] = out[y_cols].fillna(0.0)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(
        f"✅ wrote {out_parquet} with {len(out):,} rows"
        f"{'' if not skipped_no_children else f'  (skipped no-children: {skipped_no_children})'}"
        f"{'' if not skipped_empty else f'  (skipped empty actions: {skipped_empty})'}"
    )

# ------------------------------ CLI ---------------------------------

def run_from_config(cfg: Mapping[str, Any]) -> None:
    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.parquet"))
    out_pq   = Path(_get(cfg, "outputs.parquet", "data/datasets/rangenet_postflop.parquet"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    build_rangenet_postflop(manifest, out_pq, cfg)

if __name__ == "__main__":
    import argparse
    from ml.config import load_model_config  # adjust import if needed

    ap = argparse.ArgumentParser("Build RangeNet Postflop parquet from solved charts (S3-only, no cache by default)")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="Model name or YAML path (resolved by load_model_config)")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Override manifest parquet path")
    ap.add_argument("--out", type=str, default=None,
                    help="Override output parquet path")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.manifest:
        cfg.setdefault("inputs", {})["manifest"] = args.manifest
    if args.out:
        cfg.setdefault("outputs", {})["parquet"] = args.out
    run_from_config(cfg)
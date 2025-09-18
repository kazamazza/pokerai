from __future__ import annotations
import os, io, json, gzip
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import boto3
import numpy as np
import pandas as pd


# ----------------- small helpers -----------------
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

# -------- gzip-aware JSON loader (no local cache by default) --------
def _read_json_bytes(b: bytes, key_hint: str) -> dict:
    if key_hint.endswith(".gz"):
        b = gzip.GzipFile(fileobj=io.BytesIO(b)).read()
    return json.loads(b)

def _load_solver_json_from_s3(cfg: Mapping[str, Any], s3_key: str) -> dict:
    bucket = _get(cfg, "aws.bucket") or os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        raise FileNotFoundError("aws.bucket is not configured and no local cache available")
    s3 = _s3_client(cfg)
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    body = obj["Body"].read()
    # optional mirror to local cache (off by default)
    if bool(_get(cfg, "worker.cache_results", False)):
        p = Path(_get(cfg, "worker.local_cache_dir", "data/solver_cache")) / s3_key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return _read_json_bytes(body, s3_key)

# -------- 169-hand utilities --------
_RANKS = "AKQJT98765432"
def _hand_to_index_169(h: str) -> Optional[int]:
    if not h: return None
    h = h.strip()
    suited = None
    if len(h) == 2:
        r1, r2 = h[0], h[1]
    elif len(h) == 3:
        r1, r2 = h[0], h[1]; suited = h[2].lower()
    elif len(h) == 4:
        r1,s1,r2,s2 = h[0],h[1],h[2],h[3]; suited = "s" if s1==s2 else "o"
    else:
        return None
    if r1 not in _RANKS or r2 not in _RANKS: return None
    i, j = _RANKS.index(r1), _RANKS.index(r2)
    if i == j:
        row, col = i, j
    else:
        if suited == "s":
            if i > j: i, j = j, i
            row, col = j, i
        elif suited == "o":
            if i < j: i, j = j, i
            row, col = i, j
        else:
            return None
    idx = row * 13 + col
    return idx if 0 <= idx < 169 else None

def _range_to_vec169(rmap: Dict[str, float]) -> np.ndarray:
    v = np.zeros(169, dtype=np.float32)
    for h, w in (rmap or {}).items():
        idx = _hand_to_index_169(str(h))
        if idx is not None:
            try: v[idx] = float(w)
            except: pass
    return v

def _normalize_range_payload(payload) -> Dict[str, float]:
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items()}
    if isinstance(payload, str):
        out: Dict[str, float] = {}
        for tok in payload.replace(" ", "").split(","):
            if ":" not in tok: continue
            h, v = tok.split(":", 1)
            try: out[h] = float(v)
            except: pass
        return out
    return {}

# -------- root/action extraction (works with your solver dumps) --------
def _root_obj(js: dict, node_key: str = "root") -> dict:
    nodes = js.get("nodes")
    if isinstance(nodes, dict) and node_key in nodes and isinstance(nodes[node_key], dict):
        return nodes[node_key]
    if isinstance(js.get("root"), dict): return js["root"]
    if isinstance(js.get("tree"), dict): return js["tree"]
    return js

def _actions_on_node(node: dict) -> List[str]:
    for k in ("actions","action_labels","action_list"):
        arr = node.get(k)
        if isinstance(arr, list) and all(isinstance(x,str) for x in arr):
            return arr
    ch = node.get("childrens") or node.get("children") or node.get("edges")
    if isinstance(ch, dict): return list(ch.keys())
    return []

def _strategy_map(node: dict) -> Optional[dict]:
    strat = node.get("strategy")
    if isinstance(strat, dict) and "strategy" in strat and isinstance(strat["strategy"], dict):
        return strat["strategy"]
    if isinstance(strat, dict) and all(isinstance(v, (list,tuple)) for v in strat.values()):
        return strat
    return None

def _find_action_index(action_names: List[str], prefix: str) -> Optional[int]:
    pref = (prefix or "").upper()
    for i, nm in enumerate(action_names or []):
        if isinstance(nm, str) and nm.upper().startswith(pref):
            return i
    # relaxed contains
    for i, nm in enumerate(action_names or []):
        if isinstance(nm, str) and pref in nm.upper():
            return i
    return None

def _extract_action_vector_169(js: dict, *, node_key: str, action_prefix: str) -> Optional[np.ndarray]:
    node = _root_obj(js, node_key=node_key)
    strat = _strategy_map(node)
    if strat:
        names = _actions_on_node(node)
        aidx = _find_action_index(names, action_prefix)
        if aidx is not None:
            v = np.zeros(169, dtype=np.float32)
            for h, probs in strat.items():
                idx = _hand_to_index_169(h)
                if idx is None: continue
                try:
                    if isinstance(probs, (list,tuple)) and len(probs) > aidx:
                        p = float(probs[aidx])
                        if p >= 0.0 and p <= 1.0:
                            v[idx] = p
                except: pass
            if np.any(v):
                return v
    # child-node fallback: often child strategy encodes “continue” mixes
    ch = _root_obj(js, node_key=node_key).get("childrens") or {}
    for k, child in (ch.items() if isinstance(ch, dict) else []):
        if isinstance(k, str) and action_prefix.upper() in k.upper() and isinstance(child, dict):
            cstrat = _strategy_map(child)
            if cstrat:
                v = np.zeros(169, dtype=np.float32)
                # pick second component if present (after the action choice)
                for h, probs in cstrat.items():
                    idx = _hand_to_index_169(h)
                    if idx is None: continue
                    try:
                        if isinstance(probs, (list,tuple)):
                            j = 1 if len(probs) > 1 else (len(probs) - 1)
                            p = float(probs[j])
                            if p >= 0.0 and p <= 1.0:
                                v[idx] = p
                    except: pass
                if np.any(v):
                    return v
    return None

# ---------- actor / action decision (in sync with your menus) ----------
def _role_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[1] if "." in m else ""

def _donk_available(menu_id: str, actor: str) -> bool:
    grp = (menu_id or "").split(".", 1)[0]
    role = _role_from_menu(menu_id)
    actor = (actor or "").lower()
    if grp.startswith("limped_multi"):
        return False
    if actor != "oop":
        return False
    return ("Caller_OOP" in role) or grp.startswith("limped_single")

def _infer_actor(row: pd.Series) -> str:
    vpos = str(row.get("villain_pos", "")).upper()
    if vpos in ("IP", "OOP"):
        return vpos.lower()
    actor = str(row.get("actor", "")).lower()
    if actor in ("ip", "oop"):
        return actor
    pos = str(row.get("positions", "")).upper()
    if pos.startswith("IPV"): return "ip"
    if pos.startswith("OOPV"): return "oop"
    return "ip"

def _resolve_actor_action(row: pd.Series) -> Tuple[str, str]:
    actor = _infer_actor(row)                  # 'ip' or 'oop'
    menu_id = str(row.get("bet_sizing_id", ""))
    action = "DONK" if _donk_available(menu_id, actor) else "BET"
    return actor, action

def _split_positions(positions: str) -> Tuple[str,str]:
    s = str(positions or "")
    if "v" in s:
        a,b = s.split("v",1)
        return a.upper(), b.upper()
    if "V" in s:
        a,b = s.split("V",1)
        return a.upper(), b.upper()
    return ("IP","OOP")

# ---------- core builder ----------
def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Mapping[str, Any],
) -> None:
    """
    Read manifest → load each JSON → extract (actor, action)-conditioned 169 vector at root
    → group/average per scenario → write parquet that matches PostflopRangeDatasetParquet.
    """
    df = pd.read_parquet(manifest_path)

    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))
    normalize_y = bool(_get(cfg, "dataset.normalize_labels", True))

    # required manifest columns for this builder
    need = ["s3_key","positions","street","effective_stack_bb","pot_bb","bet_sizing_id","ctx"]
    need += ["board_cluster_id"] if use_clusters else ["board"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    # buckets keyed by scenario (aligned to dataset DEFAULT_X)
    buckets: Dict[Tuple, List[np.ndarray]] = {}
    weights: Dict[Tuple, float] = {}
    skipped_empty = 0

    for _, r in df.iterrows():
        s3_key = str(r["s3_key"])
        js = _load_solver_json_from_s3(cfg, s3_key)

        actor, action = _resolve_actor_action(r)
        node_key = str(r.get("node_key") or "root")

        # 1) preferred: action-conditioned vector at root
        vec = _extract_action_vector_169(js, node_key=node_key, action_prefix=action)

        # 2) fallback: raw actor range on root (rare in your dumps)
        if vec is None or not np.any(vec):
            root = _root_obj(js, node_key=node_key)
            rmap = {}
            if isinstance(root.get("ranges"), dict) and actor in root["ranges"]:
                rmap = _normalize_range_payload(root["ranges"][actor])
            elif isinstance(root.get("actors"), dict) and isinstance(root["actors"].get(actor), dict):
                rmap = _normalize_range_payload(
                    root["actors"][actor].get("range") or root["actors"][actor].get("ranges")
                )
            vec = _range_to_vec169(rmap)

        if not np.any(vec):
            skipped_empty += 1
            continue

        if normalize_y:
            s = float(vec.sum())
            if s > 0:
                vec = vec / s

        ip_seat, oop_seat = _split_positions(r["positions"])
        hero_pos = ip_seat if actor == "ip" else oop_seat

        key = (
            int(round(float(r["effective_stack_bb"]))),    # stack_bb
            float(r["pot_bb"]),                            # pot_bb
            hero_pos,                                      # hero_pos
            ip_seat,                                       # ip_pos
            oop_seat,                                      # oop_pos
            int(r["street"]),                              # street
            str(r["ctx"]),                                 # ctx
            int(r["board_cluster_id"]) if use_clusters else str(r["board"]),  # board_cluster/board
        )

        buckets.setdefault(key, []).append(vec.astype("float32"))
        weights[key] = weights.get(key, 0.0) + 1.0

    # materialize rows (schema aligned)
    rows: List[Dict[str, Any]] = []
    for key, vecs in buckets.items():
        (stack_bb, pot_bb, hero_pos, ip_pos, oop_pos, street, ctx, board_key) = key
        y = np.mean(np.stack(vecs, axis=0), axis=0)
        row = {
            "stack_bb": int(stack_bb),
            "pot_bb": float(pot_bb),
            "hero_pos": str(hero_pos),
            "ip_pos": str(ip_pos),
            "oop_pos": str(oop_pos),
            "street": int(street),
            "ctx": str(ctx),
            "weight": float(weights[key]),
        }
        if use_clusters:
            row["board_cluster"] = int(board_key)
            row["board_cluster_id"] = int(board_key)  # keep both for convenience
        else:
            row["board"] = str(board_key)

        for i, v in enumerate(y.tolist()):
            row[f"y_{i}"] = float(v)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(f"✅ wrote {out_parquet} with {len(out):,} rows"
          f"{'' if not skipped_empty else f'  (skipped empty vectors: {skipped_empty})'}")

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
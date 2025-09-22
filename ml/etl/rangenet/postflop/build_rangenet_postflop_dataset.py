from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_model_config
from infra.storage.s3_client import S3Client
from ml.range.solvers.utils.solver_json_extract import hand_to_index_169
import io
import gzip
import json
import time
import random
from pathlib import Path
from typing import Mapping, Any, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from botocore.exceptions import ClientError, BotoCoreError

# --------- tiny cfg + retry helpers ----------
def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _retry(fn, *, tries: int = 5, jitter: float = 0.25, base_sleep: float = 0.6):
    last = None
    for i in range(tries):
        try:
            return fn()
        except (ClientError, BotoCoreError) as e:
            last = e
            sleep = base_sleep * (2 ** i) + random.random() * jitter
            time.sleep(sleep)
    if last:
        raise last

# --------- local cache + loader ----------
def _cache_root(cfg: Mapping[str, Any]) -> Path:
    return Path(_get(cfg, "worker.local_cache_dir", "data/solver_cache"))

def _cache_path_for_key(cfg: Mapping[str, Any], s3_key: str) -> Path:
    return (_cache_root(cfg) / s3_key).resolve()

def _read_json_file_allow_gz(p: Path) -> dict:
    b = p.read_bytes()
    if p.suffix == ".gz":
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
            text = gz.read().decode("utf-8")
        return json.loads(text)
    else:
        return json.loads(b.decode("utf-8"))

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3c: S3Client, s3_key: str) -> tuple[dict, bool]:
    """
    Returns (json_obj, cache_hit).
    1) Try local cache; 2) download from S3 with retries; 3) read again from cache.
    """
    local_path = _cache_path_for_key(cfg, s3_key)
    if local_path.is_file():
        return _read_json_file_allow_gz(local_path), True

    def _dl():
        s3c.download_file_if_missing(s3_key, local_path)
        return True

    _retry(_dl, tries=5, jitter=0.3, base_sleep=0.8)
    return _read_json_file_allow_gz(local_path), False

# --------- minimal menu/actor/labels helpers ----------
def _role_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[1] if "." in m else ""

def _split_positions(positions: str) -> Tuple[str, str]:
    s = str(positions).upper()
    if "V" in s:
        a, b = s.split("V", 1)
        return a, b
    return ("IP","OOP")

def _parse_child_label(label: str) -> tuple[str, Optional[int], str]:
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
    strat_map = (root.get("strategy") or {}).get("strategy") or {}
    if not strat_map:
        return None
    v = np.zeros(169, dtype=np.float32)
    any_set = False
    for hand, probs in strat_map.items():
        idx = hand_to_index_169(str(hand))  # assumes available in your codebase
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

def _dedupe_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    before = len(df)
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")
    df2 = df.drop_duplicates(subset=["s3_key", "node_key"]).reset_index(drop=True)
    after = len(df2)
    return df2, before, after

# --------- pick root (delegates to your schema) ----------
def pick_root(js: dict, node_key: str = "root") -> dict:
    # If your JSON schema differs, point this to your existing util.
    # Common pattern:
    # - Monker-like export: js["nodes"][node_key]
    # - Flat: js if contains "childrens", else js["root"]
    if isinstance(js, dict) and "childrens" in js and "strategy" in js:
        return js
    if "root" in js:
        return js["root"]
    # fallback to first dict that looks like a node
    for v in js.values():
        if isinstance(v, dict) and "childrens" in v:
            return v
    return js

# --------- core builder ----------
def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Mapping[str, Any],
    *,
    checkpoint_every: Optional[int] = None,
    checkpoint_s3_prefix: Optional[str] = None,
) -> None:
    df = pd.read_parquet(manifest_path)

    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))
    normalize_y = bool(_get(cfg, "dataset.normalize_labels", True))

    need = ["s3_key", "positions", "street", "effective_stack_bb", "pot_bb", "bet_sizing_id", "ctx"]
    need += ["board_cluster_id"] if use_clusters else ["board"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    df, m_before, m_after = _dedupe_manifest(df)
    if m_before != m_after:
        print(f"⚠️ manifest deduped: {m_before} → {m_after}")

    s3c = S3Client()

    rows: List[Dict[str, Any]] = []
    skipped_empty = 0
    skipped_no_children = 0
    cache_hits = 0
    fetched = 0

    total = len(df)
    for idx, (_, r) in enumerate(tqdm(df.iterrows(), total=total, desc="Building postflop dataset")):
        s3_key  = str(r["s3_key"])
        node_key = str(r.get("node_key") or "root")

        js, cache_hit = _load_solver_json_local_or_s3(cfg, s3c, s3_key)
        cache_hits += int(cache_hit)
        fetched    += int(not cache_hit)

        root = pick_root(js, node_key=node_key)
        childrens = root.get("childrens") or {}
        child_labels = list(childrens.keys())
        if not child_labels:
            skipped_no_children += 1
            continue

        ip_pos, oop_pos = _split_positions(str(r["positions"]))
        stack_bb  = int(round(float(r["effective_stack_bb"])))
        pot_bb    = float(r["pot_bb"])
        street    = int(r["street"])
        ctx       = str(r["ctx"])
        board_key = (int(r["board_cluster_id"]) if use_clusters else str(r["board"]))
        menu_id   = str(r["bet_sizing_id"])

        role = _role_from_menu(menu_id).upper()
        actor = ("ip" if role.endswith("_IP") else
                 "oop" if role.endswith("_OOP") else "ip")
        hero_pos = ip_pos if actor == "ip" else oop_pos

        for ci, lab in enumerate(child_labels):
            action, size_pct, _ = _parse_child_label(lab)
            vec = _extract_child_vec_169(root, ci)
            if (vec is None) or (not np.any(vec)):
                skipped_empty += 1
                continue

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
                "actor": actor,
                "action": action,
                "bet_size_pct": size_pct,
                "node_key": node_key,
                "weight": 1.0,
            }
            if use_clusters:
                row["board_cluster"] = int(board_key)
                row["board_cluster_id"] = int(board_key)
            else:
                row["board"] = str(board_key)

            for i, v in enumerate(vec.tolist()):
                row[f"y_{i}"] = float(v)

            rows.append(row)

        # optional checkpointing
        if checkpoint_every and len(rows) and (idx + 1) % checkpoint_every == 0:
            ck = pd.DataFrame(rows)
            y_cols = [c for c in ck.columns if c.startswith("y_")]
            ck[y_cols] = ck[y_cols].fillna(0.0)
            ck_path = out_parquet.with_suffix(f".part_{idx+1}.parquet")
            ck.to_parquet(ck_path, index=False)
            if checkpoint_s3_prefix:
                s3_key_ck = f"{checkpoint_s3_prefix}/{ck_path.name}"
                _retry(lambda: S3Client().upload_file(ck_path, s3_key_ck), tries=4, base_sleep=0.6, jitter=0.2)

    out = pd.DataFrame(rows)
    if len(out):
        y_cols = [c for c in out.columns if c.startswith("y_")]
        out[y_cols] = out[y_cols].fillna(0.0)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    # Optional S3 upload of the final parquet
    s3_out_key = _get(cfg, "outputs.s3_key")
    if s3_out_key:
        _retry(lambda: s3c.upload_file(out_parquet, s3_out_key), tries=5, base_sleep=0.8, jitter=0.3)

    print(
        f"✅ wrote {out_parquet} with {len(out):,} rows"
        f"{'' if not skipped_no_children else f'  (skipped no-children: {skipped_no_children})'}"
        f"{'' if not skipped_empty else f'  (skipped empty actions: {skipped_empty})'}"
        f"  [manifest {m_before}→{m_after}, cache_hits={cache_hits}, fetched={fetched}]"
    )

def run_from_config(cfg: Mapping[str, Any], *, checkpoint_every: Optional[int], checkpoint_s3_prefix: Optional[str]) -> None:
    # Hardcoded output locations
    LOCAL_OUT = Path("data/datasets/rangenet_postflop.parquet")
    S3_OUT_KEY = "datasets/rangenet_postflop.parquet"

    # Manifest still comes from config (or default)
    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.dedup.parquet"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    # Build dataset (writes local parquet)
    build_rangenet_postflop(
        manifest_path=manifest,
        out_parquet=LOCAL_OUT,
        cfg=cfg,
        checkpoint_every=checkpoint_every,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
    )

    # Upload final parquet to S3 (hardcoded key)
    s3c = S3Client()
    s3c.upload_file(LOCAL_OUT, S3_OUT_KEY)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Build RangeNet Postflop parquet from solved charts (S3-resilient)")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="Model name or YAML path")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="Write partial parquet every N manifest rows (and optionally upload)")
    ap.add_argument("--checkpoint-s3-prefix", type=str, default=None,
                    help="If set, upload each partial parquet to this S3 prefix")
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    run_from_config(
        cfg,
        checkpoint_every=args.checkpoint_every,
        checkpoint_s3_prefix=args.checkpoint_s3_prefix,
    )
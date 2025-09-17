# ml/range/solvers/parser.py

from __future__ import annotations
import gzip, json, hashlib
from pathlib import Path
from typing import Dict, Optional, Any
from infra.storage.s3_client import S3Client
from ml.config.types_hands import RANK_TO_I, HAND_TO_ID
from ml.range.solvers.keying import solve_sha1

# ------------------------------
# JSON parsing → 169 distribution
# ------------------------------

def _walk_to_node(root: dict, node_key: str) -> Optional[dict]:
    """
    node_key like "root/BET 50/CALL" → traverse 'childrens'.
    If node_key == "root" or empty → return root.
    """
    if not node_key or node_key.lower() == "root":
        return root
    cur = root
    for name in node_key.split("/"):
        if not name or name.lower() == "root":
            continue
        ch = cur.get("childrens")
        if not isinstance(ch, dict) or name not in ch:
            return None
        cur = ch[name]
    return cur

def _combo_to_169(combo: str) -> Optional[str]:
    """
    "AsAh" -> "AA"
    "AhKh" -> "AKs"
    "AhKd" -> "AKo"
    """
    combo = combo.strip()
    if len(combo) != 4:
        return None
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    # order ranks high->low per your RANK_TO_I ordering
    if RANK_TO_I[r1] > RANK_TO_I[r2]:
        hi, lo, sh, sl = r1, r2, s1, s2
    else:
        hi, lo, sh, sl = r2, r1, s2, s1
    if hi == lo:
        return hi + lo
    suited = (sh == sl)
    return f"{hi}{lo}{'s' if suited else 'o'}"

def extract_range_map(data: dict, *, actor: str, node_key: str) -> Dict[str, float]:
    """
    Core extractor: walks to node_key, reads strategy, compresses to 169 codes, renormalizes.
    Returns {hand169_code -> prob} for the requested actor at the node.
    """
    node = _walk_to_node(data, node_key)
    if not node:
        return {}

    # strategy can be:
    #  - node["strategy"]["strategy"] = { "AsAh": [p1, p2, ...], ... }
    #  - node["strategy"][actor]["strategy"] = { ... }
    strat = None
    st = node.get("strategy")
    if isinstance(st, dict):
        if "strategy" in st and isinstance(st["strategy"], dict):
            strat = st["strategy"]
        elif actor in st and isinstance(st[actor], dict) and isinstance(st[actor].get("strategy"), dict):
            strat = st[actor]["strategy"]

    if not isinstance(strat, dict):
        return {}

    accum: Dict[str, float] = {}
    total = 0.0
    for combo_str, action_probs in strat.items():
        try:
            w = float(sum(action_probs))  # presence weight ~ sum of action probs
        except Exception:
            continue
        code169 = _combo_to_169(combo_str)
        if not code169 or code169 not in HAND_TO_ID:
            continue
        accum[code169] = accum.get(code169, 0.0) + w
        total += w

    if total <= 0:
        return {}

    inv = 1.0 / total
    for k in list(accum.keys()):
        accum[k] *= inv
    return accum

def parse_solver_json(path: Path, *, actor: str, node_key: str) -> Dict[str, float]:
    """
    Accept .json or .json.gz | return {169_code -> prob}
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return extract_range_map(data, actor=actor, node_key=node_key)


# ------------------------------
# Cache-first loader
# ------------------------------

def try_load_solved_from_cache(
    cfg: Dict[str, Any],
    params: Dict[str, Any],   # canonicalized inputs
    local_cache_dir: Path,     # e.g. data/solver_cache
) -> Path | None:
    """
    Look for local cache (json or json.gz). If not found and upload_s3=true, pull from S3 into local gz.
    Return local path (either .json or .json.gz) or None if not found.
    """
    h = solve_sha1(params)
    rel = f"{h}/output_result.json"
    local_json = local_cache_dir / rel
    local_gz   = local_cache_dir / f"{h}/output_result.json.gz"

    if local_json.exists():
        return local_json
    if local_gz.exists():
        return local_gz

    # S3 lookup & download
    if cfg.get("worker", {}).get("upload_s3", False):
        s3 = S3Client(bucket_name=cfg["worker"].get("s3_bucket"))
        prefix = cfg["worker"].get("s3_prefix", "worker/outputs/v1")
        s3_key = s3_key_for_solve(params, h, prefix=prefix)
        try:
            local_gz.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(s3_key, local_gz)
            return local_gz
        except Exception:
            return None

    return None


def load_villain_range_cached_only(
    cfg: Dict[str, Any],
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str | None,
    board_cluster_id: int | None,
    range_ip: str,
    range_oop: str,
    positions: str,         # "OOPvIP" or "IPvOOP"
    street: int,            # 1/2/3 flop/turn/river
    bet_sizing_id: str,     # e.g., "std" (used in manifest → hashed into params)
    accuracy: float,
    max_iter: int,
    allin_threshold: float,
    actor: str,             # "ip" or "oop"
    node_key: str = "root", # use "root" unless you want a specific child path
    local_cache_dir: str | Path = "data/solver_cache",
) -> Dict[str, float]:
    """
    Strict cache/read: never runs the worker.
    Expects the solve to be precomputed and present locally or in S3.
    """
    params = {
        "street": street,
        "pot_bb": float(pot_bb),
        "effective_stack_bb": float(effective_stack_bb),
        "board": board or "",
        "board_cluster_id": int(board_cluster_id) if board_cluster_id is not None else None,
        "range_ip": range_ip,
        "range_oop": range_oop,
        "positions": positions,
        "bet_sizing_id": bet_sizing_id,
        "accuracy": float(accuracy),
        "max_iter": int(max_iter),
        "allin_threshold": float(allin_threshold),
        # include anything else that changes the tree (worker version, menus hash, etc.)
        "solver_version": cfg.get("worker", {}).get("version", "v1"),
    }

    local_cache_dir = Path(local_cache_dir)
    hit = try_load_solved_from_cache(cfg, params, local_cache_dir)
    if not hit:
        # In prod you usually return {} (and log) or raise to catch coverage holes
        raise FileNotFoundError("No cached solve found for params; pre-solve first.")
    return parse_solver_json(hit, actor=actor, node_key=node_key)
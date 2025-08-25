# ml/range/solvers/adapter_cache_first.py
from pathlib import Path
from typing import Dict, Any
import gzip, json

from infra.storage.s3_client import S3Client
from ml.range.solvers.keying import solve_sha1, s3_key_for_solve

def try_load_solved_from_cache(
    cfg: Dict[str, Any],
    params: Dict[str, Any],   # canonicalized inputs
    local_cache_dir: Path,     # e.g. data/solver_cache
) -> Path | None:
    sha1 = solve_sha1(params)
    rel  = f"{sha1}/output_result.json"  # local layout
    local_json = (local_cache_dir / rel)
    local_gz   = local_json.with_suffix(".json.gz")
    if local_json.exists():
        return local_json
    if local_gz.exists():
        # optional: return gz path and let parser read gz
        return local_gz

    # S3 lookup
    if cfg["solver"].get("upload_s3", False):
        s3  = S3Client(bucket_name=cfg["solver"].get("s3_bucket"))
        s3_key = s3_key_for_solve(params, sha1, prefix=cfg["solver"].get("s3_prefix","solver/outputs/v1"))
        # Download gz into cache
        local_gz.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(s3_key, local_gz)
            return local_gz
        except Exception:
            return None
    return None

def parse_solver_json(path: Path, actor: str, node_key: str) -> Dict[str, float]:
    # accept .json or .json.gz
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    else:
        data = json.loads(path.read_text())
    # TODO: your existing parse logic that extracts actor@node_key → {hand:prob}
    return extract_range_map(data, actor=actor, node_key=node_key)

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
    street: int,            # 1/2/3 flop/turn/river (if your solver supports)
    bet_sizing_id: str,     # "std" etc.
    accuracy: float,
    max_iter: int,
    allin_threshold: float,
    actor: str,
    node_key: str = "flop_root",
    local_cache_dir: str | Path = "data/solver_cache",
) -> Dict[str, float]:
    params = {
        "street": street,
        "pot_bb": pot_bb,
        "effective_stack_bb": effective_stack_bb,
        "board": board,
        "board_cluster_id": board_cluster_id,
        "range_ip": range_ip,
        "range_oop": range_oop,
        "positions": positions,
        "bet_sizing_id": bet_sizing_id,
        "accuracy": accuracy,
        "max_iter": max_iter,
        "allin_threshold": allin_threshold,
    }
    local_cache_dir = Path(local_cache_dir)
    hit = try_load_solved_from_cache(cfg, params, local_cache_dir)
    if not hit:
        # In prod: either return {} or raise; during bring‑up you may fallback to on‑demand:
        # return {}
        raise FileNotFoundError("No cached solve found for params; pre-solve first.")
    return parse_solver_json(hit, actor=actor, node_key=node_key)
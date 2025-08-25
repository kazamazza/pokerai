from pathlib import Path
from typing import Dict, Any
from ml.range.solvers.parser import try_load_solved_from_cache, parse_solver_json


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
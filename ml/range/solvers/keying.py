import json, hashlib
from typing import Dict, Any

CANON_FIELDS = [
    "street", "pot_bb", "effective_stack_bb",
    "board",                   # or "board_cluster_id"
    "range_ip", "range_oop",
    "positions",               # e.g. "OOPvIP"
    "bet_sizing_id",           # your abstraction tag
    "accuracy", "max_iter", "allin_threshold",
]

def canonical_payload(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pick + order only fields that define a unique solve."""
    payload = {k: params[k] for k in CANON_FIELDS if k in params}
    return payload

def solve_sha1(params: Dict[str, Any]) -> str:
    payload = canonical_payload(params)
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

def s3_key_for_solve(params: Dict[str, Any], sha1: str, prefix: str = "worker/outputs/v1") -> str:
    street = params.get("street", 1)
    pos    = params.get("positions", "OOPvIP")
    stack  = int(round(float(params.get("effective_stack_bb", 100))))
    pot    = int(round(float(params.get("pot_bb", 10))))
    board  = params.get("board") or f"cluster_{params.get('board_cluster_id')}"
    acc    = str(params.get("accuracy", 0.5))
    sizes  = params.get("bet_sizing_id", "std")
    return f"{prefix}/street={street}/pos={pos}/stack={stack}/pot={pot}/board={board}/acc={acc}/sizes={sizes}/{sha1}/output_result.json.gz"
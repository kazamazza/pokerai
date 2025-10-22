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


def _sanitize_board(board: str | None) -> str:
    if not board:
        return ""
    return str(board).replace(",", "").replace(" ", "")

def s3_key_base(
    params: Dict[str, Any],
    sha: str,
    prefix: str = "solver/outputs/v1",
) -> str:
    street = int(params.get("street", 1))
    pos = str(params.get("positions", "UNKvUNK"))
    stack = int(round(float(params.get("effective_stack_bb", 0))))
    pot = int(round(float(params.get("pot_bb", 0))))
    board = _sanitize_board(params.get("board") or f"cluster_{int(params.get('board_cluster_id', -1))}")
    acc = float(params.get("accuracy", 0.01))
    sizes = str(params.get("bet_sizing_id", "std"))
    shard = solve_sha1(params)[:2]  # same sha1, no size in params

    return (
        f"{prefix}"
        f"/street={street}"
        f"/pos={pos}"
        f"/stack={stack}"
        f"/pot={pot}"
        f"/board={board}"
        f"/acc={acc:.2f}"
        f"/sizes={sizes}"
        f"/{shard}/{sha}"
    )

def s3_key_for_size(base_key: str, size_pct: int) -> str:
    """
    Turns a base directory into a concrete object key for a given size.
    """
    return f"{base_key}/size={int(size_pct)}p/output_result.json.gz"
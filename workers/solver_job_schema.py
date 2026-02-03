# workers/solver_job_schema.py
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =========================
# Manifest schema contract
# =========================

REQUIRED_COLUMNS = {
    "sha1",
    "s3_key",
    "street",
    "pot_bb",
    "effective_stack_bb",
    "ip_pos",
    "oop_pos",
    "board",
    "board_cluster_id",
    "bet_sizing_id",
    "bet_sizes",
    "range_ip",
    "range_oop",
    "accuracy",
    "max_iter",
    "allin_threshold",
    "solver_version",
    "node_key",
}

# These are copied into msg["params"] (plus size_pct) for the solver worker.
PARAM_FIELDS = [
    "street",
    "pot_bb",
    "effective_stack_bb",
    "ip_pos",
    "oop_pos",
    "board",
    "board_cluster_id",
    "bet_sizing_id",
    "range_ip",
    "range_oop",
    "accuracy",
    "max_iter",
    "allin_threshold",
    "solver_version",
    "node_key",
]

# =========================
# Parsing helpers
# =========================

def parse_bet_sizes(cell: Any) -> List[int]:
    """
    Parse manifest bet_sizes into ordered unique integer percents.

    Accepts:
      - [0.33, 0.66]
      - [33, 66]
      - pyarrow list structs: [{"element": 0.33}, ...]
      - numpy arrays

    Returns:
      - sorted? NO: preserves input order, de-duped.
    """
    if cell is None:
        return []

    # unwrap pyarrow
    try:
        if hasattr(cell, "as_py"):
            cell = cell.as_py()
    except Exception:
        pass

    # unwrap numpy
    try:
        import numpy as np
        if isinstance(cell, np.ndarray):
            cell = cell.tolist()
    except Exception:
        pass

    seq = cell if isinstance(cell, list) else [cell]
    out: List[int] = []
    seen = set()

    for it in seq:
        if it is None:
            continue
        v = it.get("element") if isinstance(it, dict) else it
        try:
            f = float(v)
        except Exception:
            continue

        # accept ratios (<=3.0) or percents (>3.0)
        if f <= 3.0:
            pct = int(Decimal(str(f * 100)).quantize(0, ROUND_HALF_UP))
        else:
            pct = int(Decimal(str(f)).quantize(0, ROUND_HALF_UP))

        if 1 <= pct <= 200 and pct not in seen:
            out.append(pct)
            seen.add(pct)

    return out


def inject_size_into_s3_key(base_key: str, size_pct: int) -> str:
    """
    Convert a manifest base s3_key into the per-size output key.
    """
    base = str(base_key).rstrip("/")
    return f"{base}/size={int(size_pct)}p/output_result.json.gz"

def validate_manifest_columns(columns: Iterable[str]) -> Tuple[bool, List[str]]:
    cols = set(columns)
    missing = sorted(REQUIRED_COLUMNS - cols)
    return (len(missing) == 0, missing)


# =========================
# Message building
# =========================

def _jsonable_scalar(x: Any) -> Any:
    """
    Normalize pandas/pyarrow/numpy scalars into JSON-serializable primitives.
    """
    # unwrap pyarrow scalar
    try:
        if hasattr(x, "as_py"):
            x = x.as_py()
    except Exception:
        pass

    # unwrap numpy scalar
    try:
        import numpy as np
        if isinstance(x, np.generic):
            x = x.item()
    except Exception:
        pass

    return x


def normalize_params(row: Any, size_pct: int) -> Dict[str, Any]:
    """
    Build params dict from a parquet row (itertuples row),
    ensuring JSON-serializable values and consistent types.
    """
    params: Dict[str, Any] = {}
    for k in PARAM_FIELDS:
        params[k] = _jsonable_scalar(getattr(row, k))

    # enforce types
    params["street"] = int(params["street"])
    if params["street"] not in (1, 2, 3):
        raise ValueError(f"Unsupported street={params['street']} (expected 1/2/3)")

    params["pot_bb"] = float(params["pot_bb"])
    params["effective_stack_bb"] = float(params["effective_stack_bb"])
    params["board_cluster_id"] = int(params["board_cluster_id"]) if params["board_cluster_id"] is not None else None

    params["accuracy"] = float(params["accuracy"])
    params["max_iter"] = int(params["max_iter"])
    params["allin_threshold"] = float(params["allin_threshold"])

    params["ip_pos"] = str(params["ip_pos"])
    params["oop_pos"] = str(params["oop_pos"])
    params["board"] = str(params["board"])
    params["bet_sizing_id"] = str(params["bet_sizing_id"])
    params["solver_version"] = str(params["solver_version"])
    params["node_key"] = str(params["node_key"])
    params["range_ip"] = str(params["range_ip"])
    params["range_oop"] = str(params["range_oop"])

    # add size
    params["size_pct"] = int(size_pct)

    return params


def build_message(
    row: Any,
    *,
    size_pct: int,
    pilot: bool = False,
    category_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a single SQS message body dict for one manifest row + one size_pct.
    """
    params = normalize_params(row, size_pct)

    sha1 = str(_jsonable_scalar(getattr(row, "sha1")))
    base_s3_key = str(_jsonable_scalar(getattr(row, "s3_key")))
    s3_key = inject_size_into_s3_key(base_s3_key, size_pct)

    msg: Dict[str, Any] = {
        "job_id": f"{sha1}:{int(size_pct)}",
        "sha1": sha1,
        "base_s3_key": base_s3_key,
        "s3_key": s3_key,
        "params": params,
    }

    if pilot:
        msg["pilot"] = True
        if category_key is not None:
            msg["category_key"] = category_key

    return msg


def build_messages(row: Any) -> List[Dict[str, Any]]:
    """
    Expand a manifest row into N messages (one per bet size).
    """
    sizes = parse_bet_sizes(_jsonable_scalar(getattr(row, "bet_sizes")))
    if not sizes:
        sha1 = getattr(row, "sha1", "<unknown>")
        raise ValueError(f"Manifest row sha1={sha1} has no bet_sizes")

    return [build_message(row, size_pct=size_pct) for size_pct in sizes]


def validate_manifest_columns(columns: Iterable[str]) -> Tuple[bool, List[str]]:
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    return (len(missing) == 0, missing)
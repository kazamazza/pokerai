import gzip
import hashlib
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Mapping, Any

from botocore.exceptions import BotoCoreError, ClientError

from infra.storage.s3_client import S3Client


def parse_amount_from_label(up: str) -> float | None:
    # works for "BET 5.000000", "RAISE 25.000000" etc.
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", up)
    return float(m.group(1)) if m else None

def bucket_bet_pct(pct: float | None) -> str:
    if pct is None: return "BET_33"
    if pct < 30:  return "BET_25"
    if pct < 42:  return "BET_33"
    if pct < 58:  return "BET_50"
    if pct < 71:  return "BET_66"
    if pct < 90:  return "BET_75"
    return "BET_100"

def bucket_raise_x(x: float | None) -> str:
    if x is None: return "RAISE_200"
    if x < 1.75: return "RAISE_150"
    if x < 2.5:  return "RAISE_200"
    return "RAISE_300"

def bucket_bet_label(up: str, pot_bb: float, actor: str) -> str:
    # OOP betting into a check → treat as DONK family; we only keep one bucket
    if actor == "oop":
        return "DONK_33"
    amt = parse_amount_from_label(up) or 0.0
    pct = 100.0 * amt / max(pot_bb, 1e-9)
    return bucket_bet_pct(pct)

def bucket_raise_label(up: str, current_bet_bb: float) -> str:
    # map "RAISE A" using A / current_bet_bb → {150,200,300}
    amt = parse_amount_from_label(up) or 0.0
    x = amt / max(current_bet_bb, 1e-9)
    return bucket_raise_x(x)

def _get(cfg: Mapping[str, any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _stable_shard_index(s3_key: str, node_key: str, m: int) -> int:
    # stable, deterministic shard from (s3_key, node_key)
    h = hashlib.sha1(f"{s3_key}|{node_key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % m

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

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3c: S3Client, s3_key: str) -> dict:
    local_path = _cache_path_for_key(cfg, s3_key)
    if not local_path.is_file():
        _retry(lambda: s3c.download_file_if_missing(s3_key, local_path))
    return _read_json_file_allow_gz(local_path)

def _is_action_node(n: dict) -> bool:
    """Ignore chance/terminal nodes."""
    t = str(n.get("node_type", "")).lower()
    if t in {"chance_node", "terminal", "showdown"}:
        return False
    if "dealcards" in n and t == "chance_node":
        return False
    return True


def _split_positions(positions: str) -> tuple[str, str]:
    """
    Split a positions string into (ip_pos, oop_pos).
    Supports formats like:
      - "BTN_vs_BB"
      - "SB_vs_BB"
      - "srp_hu.PFR_IP" (falls back gracefully)

    Returns
    -------
    (ip_pos, oop_pos) : tuple of str
    """
    if not positions:
        return "IP", "OOP"

    txt = str(positions).strip().upper()

    # Common "X_vs_Y" pattern
    if "_VS_" in txt:
        left, right = txt.split("_VS_", 1)
        return left, right

    # Sometimes encoded as dot + role
    if "." in txt:
        # e.g. "srp_hu.PFR_IP"
        base, role = txt.split(".", 1)
        role = role.upper()
        if role.endswith("_IP"):
            return role.replace("_IP", ""), "OOP"
        elif role.endswith("_OOP"):
            return "IP", role.replace("_OOP", "")
        return role, "OOP"

    # Default fallback
    return "IP", "OOP"
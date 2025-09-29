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


NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")

def _last_number(s: str) -> float | None:
    m = NUM.findall(str(s))
    if not m: return None
    try: return float(m[-1])
    except Exception: return None

def _has_any(s: str, *subs: str) -> bool:
    u = s.upper()
    return any(x in u for x in subs)

def parse_root_bet_size_bb(root_actions: list[str], pot_bb: float) -> float | None:
    """
    Find a plausible facing bet size at root in **big blinds**.
    Handles:
      - 'BET 50%'  -> 0.50 * pot_bb
      - 'BET 3.00' -> 3.00 bb
      - 'BET 100%' -> pot_bb
      - 'BET TO 7.50' -> 7.50 bb
    Returns None if nothing found.
    """
    bet_bb_candidates = []
    for a in root_actions:
        up = str(a).upper()
        if not up.startswith("BET"):
            continue
        # percent form
        if "%" in up:
            pct = _last_number(up)
            if pct is not None and pot_bb > 0:
                bet_bb_candidates.append((pct/100.0) * pot_bb)
        else:
            # absolute amount form
            v = _last_number(up)
            if v is not None:
                bet_bb_candidates.append(v)
    if bet_bb_candidates:
        # choose the **largest** bet size at root (usually the facing bet of interest)
        return float(max(bet_bb_candidates))
    return None

def parse_raise_to_bb(raise_label: str, pot_bb: float, bet_size_bb: float | None) -> float | None:
    """
    Parse the **raise-to** amount in BB from a raise label.
    Handles forms like:
      - 'RAISE TO 9.00'
      - 'RAISE 9.00'
      - 'RERAISE TO 300%' (relative to pot)
      - 'RAISE 3x' (relative to facing bet if known)
      - 'RAISE 300%' (relative to pot)
    Returns the **raise-to** size in BB if possible.
    """
    up = str(raise_label).upper()

    # all-in special cases handled elsewhere (we'll bucket as ALLIN outside)

    # 1) '... x' multiplier form (e.g., 'RAISE 3X', 'RERAISE 1.5x')
    if "X" in up:
        mult = _last_number(up)
        if mult is not None and bet_size_bb and bet_size_bb > 0:
            return float(mult * bet_size_bb)

    # 2) percent form => % of pot_bb (e.g., 'RAISE TO 300%')
    if "%" in up:
        pct = _last_number(up)
        if pct is not None and pot_bb > 0:
            return float((pct / 100.0) * pot_bb)

    # 3) absolute form in bb (e.g., 'RAISE TO 9.00', 'RAISE 8.5')
    v = _last_number(up)
    if v is not None:
        return float(v)

    return None

def parse_amount_from_label(up: str) -> float | None:
    # works for "BET 5.000000", "RAISE 25.000000" etc.
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", up)
    return float(m.group(1)) if m else None

def _verb(s: str) -> str:
        u = str(s).strip().upper()
        return u.split()[0] if u else u

def _norm(s: str) -> tuple[str, float | None]:
        u = str(s).strip().upper()
        return _verb(u), _last_number(u)

def _resolve_child(children, action_label, eps: float = 1e-3):
    if not children: return {}
    verb_t, val_t = _norm(action_label)
    for k, v in children.items():
        if not isinstance(v, dict):
            continue
        vk, vv = _norm(k)
        if vk != verb_t:
            continue
        if val_t is None or vv is None:
            return v
        if abs(val_t - vv) <= eps:  # tolerate 175 vs 175.000000
            return v
    return {}

def _is_action_node(n: Mapping[str, Any]) -> bool:
        t = str(n.get("node_type", "")).lower()
        if t in {"chance_node", "terminal", "showdown"}: return False
        return True

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


def bucket_bet_label(up: str, *, pot_bb: float) -> str:
    """Bucket IP/OOP bets by **percent of pot** into your vocab."""
    v = _last_number(up)
    if v is None or pot_bb <= 0:
        return "BET_100"  # harmless fallback
    # If label included %, interpret as %pot; else interpret as bb and convert to %
    if "%" in up.upper():
        pct = v
    else:
        pct = (v / pot_bb) * 100.0
    if pct <= 27: return "BET_25"
    if pct <= 40: return "BET_33"
    if pct <= 58: return "BET_50"
    if pct <= 70: return "BET_66"
    if pct <= 85: return "BET_75"
    return "BET_100"

def bucket_raise_label(raise_label: str, *, pot_bb: float, facing_bet_bb: float | None, stack_bb: float | None = None) -> str:
    """
    Bucket a **raise-to** into RAISE_150 / 200 / 300 **relative to the facing bet**:
      m = raise_to_bb / facing_bet_bb
      m <= ~1.75  -> RAISE_150
      m <= ~2.5   -> RAISE_200
      else        -> RAISE_300
    'ALL-IN' is handled by caller when label includes that intent or when raise_to_bb >= stack_bb.
    """
    up = str(raise_label).upper()
    if _has_any(up, "ALLIN", "ALL-IN", "JAM"):
        return "ALLIN"

    if facing_bet_bb is None or facing_bet_bb <= 0:
        # without a facing bet, we cannot compute a sensible multiple → default to biggest bucket
        return "RAISE_300"

    raise_to_bb = parse_raise_to_bb(up, pot_bb=pot_bb, bet_size_bb=facing_bet_bb)
    if raise_to_bb is None:
        return "RAISE_300"

    # all-in by size vs stack?
    if stack_bb is not None and stack_bb > 0 and raise_to_bb >= 0.95 * stack_bb:
        return "ALLIN"

    m = raise_to_bb / facing_bet_bb
    if m <= 1.75: return "RAISE_150"
    if m <= 2.5:  return "RAISE_200"
    return "RAISE_300"

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


BET_ALIASES   = ("BET", "DONK", "PROBE")  # treat all as BET_* in vocab
RAISE_ALIASES = ("RAISE", "RE-RAISE", "RERAISE", "3BET", "4BET")  # solver variants
ALLIN_ALIASES = ("ALLIN", "ALL-IN", "ALL IN", "JAM", "SHOVE")

def is_call(up: str) -> bool:
    return up.startswith("CALL")

def is_fold(up: str) -> bool:
    return up.startswith("FOLD")

def is_check(up: str) -> bool:
    return up.startswith("CHECK")

def is_bet_like(up: str) -> bool:
    return any(up.startswith(a) for a in BET_ALIASES)

def is_raise_like(up: str) -> bool:
    return any(a in up for a in RAISE_ALIASES)

def is_allin_like(up: str) -> bool:
    return any(a in up for a in ALLIN_ALIASES)


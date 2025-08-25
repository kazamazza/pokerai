from functools import lru_cache
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, Any, Optional

from ml.etl.utils.monker_parser import load_range_file_cached


def _range_map_to_compact(rng: Dict[str, float]) -> str:
    """
    Convert a {hand_code -> weight} map into a compact string:
      AA,KK,QQ:0.5,AQs,AQo:0.25, ...
    (weights of 1.0 omit the :w)
    """
    items = []
    for k, v in sorted(rng.items()):
        if v >= 1.0 - 1e-12:
            items.append(k)
        else:
            items.append(f"{k}:{v:.6g}")
    return ",".join(items)

@lru_cache(maxsize=1)
def _load_preflop_range_index(manifest_path: str | Path) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Build an index {(stack_bb, position) -> range_map} from your Monker manifest.
    The manifest must have: stack_bb, hero_pos (or opener_pos), abs_path.
    If multiple files map to the same (stack,pos), we average and re-normalize.
    """
    mp = Path(manifest_path)
    if not mp.exists():
        raise FileNotFoundError(f"Preflop manifest not found: {mp}")

    df = pd.read_parquet(mp)
    pos_col = "hero_pos" if "hero_pos" in df.columns else ("opener_pos" if "opener_pos" in df.columns else None)
    if pos_col is None:
        raise ValueError("Manifest is missing hero_pos/opener_pos")

    need = {"stack_bb", pos_col, "abs_path"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Preflop manifest missing columns: {miss}")

    index: Dict[Tuple[int, str], Dict[str, float]] = {}
    counts: Dict[Tuple[int, str], int] = {}

    for _, r in df.iterrows():
        key = (int(r["stack_bb"]), str(r[pos_col]))
        rng_map = load_range_file_cached(Path(str(r["abs_path"])))  # -> Dict[hand_code, weight]
        if not rng_map:
            continue
        # accumulate (sum) then we’ll average
        cur = index.get(key, {})
        for k, v in rng_map.items():
            cur[k] = cur.get(k, 0.0) + float(v)
        index[key] = cur
        counts[key] = counts.get(key, 0) + 1

    # average + normalize
    for key, m in index.items():
        n = max(counts.get(key, 1), 1)
        s = sum(m.values())
        if s <= 0:
            # fallback uniform (rare)
            continue
        for k in list(m.keys()):
            m[k] = (m[k] / n)
        # renormalize to sum=1
        s2 = sum(m.values())
        if s2 > 0:
            for k in list(m.keys()):
                m[k] /= s2

    return index

def get_ranges_for_pair(
    *,
    stack_bb: float | int,
    ip: str,
    oop: str,
    cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Return compact Monker-like ranges (ip, oop) for a given stack/positions.

    Looks for a preflop manifest at cfg.inputs.preflop_manifest (optional).
    If present, tries exact stack match first; if not, picks nearest stack.
    If positions are abstract ("IP","OOP"), skips manifest lookup and uses defaults.
    Falls back to cfg.defaults.range_ip / cfg.defaults.range_oop.
    """
    stack_bb = int(round(float(stack_bb)))

    # Top-level (no wrapper) access
    inputs   = cfg.get("inputs", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    preflop_manifest = inputs.get("preflop_manifest")

    index: Optional[Dict[Tuple[int, str], Dict[str, float]]] = None
    if preflop_manifest:
        index = _load_preflop_range_index(preflop_manifest)  # assumed existing util

    # If caller passed abstract roles, manifest positions likely won't match → use defaults
    abstract_roles = {ip.upper(), oop.upper()} <= {"IP", "OOP"}

    def _fetch_from_index(pos: str) -> Optional[str]:
        if not index or abstract_roles:
            return None
        # Exact stack first
        hit = index.get((stack_bb, pos))
        if hit:
            return _range_map_to_compact(hit)

        # Nearest-stack fallback
        # collect all stacks available for this pos
        candidates = [s for (s, p) in index.keys() if p == pos]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda s: abs(s - stack_bb))
        hit = index.get((nearest, pos))
        if not hit:
            return None
        return _range_map_to_compact(hit)

    rng_ip  = _fetch_from_index(ip)
    rng_oop = _fetch_from_index(oop)

    # Fallback to defaults if missing
    if rng_ip is None:
        rng_ip = defaults.get("range_ip")
    if rng_oop is None:
        rng_oop = defaults.get("range_oop")

    if rng_ip is None or rng_oop is None:
        raise RuntimeError(
            f"No preflop ranges found for stack={stack_bb}, ip={ip}, oop={oop} "
            f"and no defaults provided in cfg.defaults"
        )

    return rng_ip, rng_oop
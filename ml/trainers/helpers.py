import itertools
import json
import re
from pathlib import Path
from typing import Optional


def _get(cfg, path, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur: return default
        cur = cur[p]
    return cur

def _set(cfg, path, value):
    cur = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _slug(v):
    if isinstance(v, (list, tuple)):
        v = "-".join(str(x) for x in v)
    s = str(v).replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s[:64]

def _parse_val_kl_from_ckpt(path: str) -> Optional[float]:
    """
    Prefer reading best_meta.json sitting next to the ckpt. Fallback to filename.
    """
    p = Path(path)
    meta = p.parent / "best_meta.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            v = data.get("best_val_kl", None)
            if v is not None:
                return float(v)
        except Exception:
            pass

    # Filename fallback: expects ...-{val_kl:.4f}.ckpt at the end
    # Example: rangenet-preflop-14-1.0037.ckpt  ->  1.0037
    m = re.search(r"-([0-9]*\.[0-9]+)\.ckpt$", p.name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _trial_dir(base_dir: Path, trial_params: dict) -> Path:
    # Make a readable subdir name from a few keys
    keys = ["model.hidden_dims","model.lr","model.dropout","model.label_smoothing","train.batch_size","seed"]
    parts = []
    for k in keys:
        v = trial_params.get(k, None)
        if v is not None:
            parts.append(f"{k.split('.')[-1]}={_slug(v)}")
    name = "trial__" + "__".join(parts) if parts else "trial"
    return base_dir / name

def _expand_grid(space: dict) -> list[dict]:
    """Turn a dict of dotted_key -> list(values) into list of dicts (grid)."""
    items = [(k, v if isinstance(v, list) else [v]) for k, v in space.items() if k not in ("max_trials", "sample")]
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    trials = []
    for combo in itertools.product(*vals):
        t = {k: combo[i] for i, k in enumerate(keys)}
        trials.append(t)
    # optionally truncate
    mt = space.get("max_trials", None)
    if isinstance(mt, int) and mt > 0 and len(trials) > mt:
        trials = trials[:mt]
    return trials
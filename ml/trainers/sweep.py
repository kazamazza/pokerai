# ml/train_utils/sweep.py
from __future__ import annotations
import itertools, json, shutil
from pathlib import Path
from typing import Callable, Mapping, Any, Sequence

def expand_grid(sweep: Mapping[str, Sequence[Any]], max_trials: int | None = None) -> list[dict]:
    """Cartesian product over the sweep dict (dotted keys allowed, auto-wrap scalars)."""
    import collections.abc, itertools

    keys = list(sweep.keys())
    values = []
    for v in sweep.values():
        # Treat single scalars as one-element lists
        if isinstance(v, str):
            values.append([v])
        elif isinstance(v, collections.abc.Iterable):
            try:
                # convert to list but avoid treating scalars like int/float as iterables
                if isinstance(v, (list, tuple, set)):
                    values.append(list(v))
                else:
                    # handle e.g. numpy arrays or pandas objects
                    values.append(list(v))
            except Exception:
                values.append([v])
        else:
            values.append([v])

    # flatten safely
    trials = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if max_trials is not None:
        trials = trials[: int(max_trials)]
    return trials

def apply_dotted(cfg: dict, params: Mapping[str, Any]) -> dict:
    """Set dotted keys like 'model.lr' into nested dict cfg (copy)."""
    import copy
    out = copy.deepcopy(cfg)
    for k, v in params.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out

def trial_dir(base: Path, params: Mapping[str, Any]) -> Path:
    """Stable folder name for a param combo (sorted keys, short)."""
    safe = []
    for k in sorted(params.keys()):
        v = params[k]
        val = str(v).replace("/", "-")
        safe.append(f"{k.replace('.','_')}={val}")
    name = "__".join(safe)[:180]  # guard path length
    return base / name

def parse_scalar_from_ckpt(ckpt_path: str | Path, default: float = float("inf")) -> float:
    """
    Generic 'score' parser:
    - prefer '<ckpt>.metrics.json' with {'monitor': value}
    - else try to parse from filename pattern '...-{val:.4f}.ckpt'
    - else default
    """
    p = Path(ckpt_path)
    # 1) sidecar metrics next to ckpt
    mjson = p.with_suffix(p.suffix + ".metrics.json")
    if mjson.exists():
        try:
            js = json.loads(mjson.read_text())
            # try a few common keys
            for key in ("val_loss", "val_kl", "monitor", "score"):
                if key in js:
                    return float(js[key])
        except Exception:
            pass
    # 2) suffix in filename
    try:
        stem = p.stem  # e.g., 'equitynet-03-0.1234'
        tail = stem.split("-")[-1]
        return float(tail)
    except Exception:
        return default

def finalize_best_artifacts(best_ckpt: Path, dest_dir: Path) -> None:
    """
    Copy best.ckpt, best_sidecar.json, and best_config.json into dest_dir.
    Looks for sidecar as '<ckpt>.sidecar.json' (preferred) with fallbacks.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = Path(best_ckpt)
    trial_dir = best_ckpt.parent

    # checkpoint
    shutil.copy2(best_ckpt, dest_dir / "best.ckpt")

    # sidecar: preferred '<ckpt>.sidecar.json'
    sidecar_ckpt = Path(str(best_ckpt) + ".sidecar.json")
    sidecar_trial = trial_dir / "sidecar.json"
    sidecar_best  = trial_dir / "best_sidecar.json"
    if sidecar_ckpt.exists():
        shutil.copy2(sidecar_ckpt, dest_dir / "best_sidecar.json")
    elif sidecar_trial.exists():
        shutil.copy2(sidecar_trial, dest_dir / "best_sidecar.json")
    elif sidecar_best.exists():
        shutil.copy2(sidecar_best, dest_dir / "best_sidecar.json")
    else:
        print("⚠️  Sidecar not found next to best.ckpt; ensure your trainer writes it.")

    # config snapshot (optional)
    cfg_src = trial_dir / "config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, dest_dir / "best_config.json")

def run_sweep(
    *,
    base_cfg: dict,
    sweep: dict,
    run_fn: Callable[[dict], str],           # returns ckpt path
    score_fn: Callable[[str], float],        # ckpt path -> scalar (lower is better)
    base_ckpt_dir: Path,
    monitor: str = "val_loss",
    max_trials: int | None = None,
) -> dict:
    """
    Generic grid sweep orchestrator.
    Returns {'best': {...}, 'results': [...]}
    """
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)
    trials = expand_grid(sweep, max_trials=max_trials)
    results = []
    best = {"score": float("inf"), "ckpt": None, "params": None}

    import pandas as pd
    for i, params in enumerate(trials, 1):
        tcfg = apply_dotted(base_cfg, params)
        tdir = trial_dir(base_ckpt_dir, params)
        tcfg.setdefault("train", {})["checkpoints_dir"] = str(tdir)
        Path(tdir).mkdir(parents=True, exist_ok=True)

        print(f"\n🧪 Trial {i}/{len(trials)} → {json.dumps(params)}")
        ckpt_path = run_fn(tcfg)
        score = score_fn(ckpt_path)

        row = {"trial": i, "score": score, "ckpt": ckpt_path, **params}
        results.append(row)
        pd.DataFrame(results).to_csv(base_ckpt_dir / "sweep_results.csv", index=False)

        if score < best["score"]:
            best = {"score": score, "ckpt": ckpt_path, "params": params}
        print(f"   ⮑ {monitor}={score:.6f}  best_so_far={best['score']:.6f}")

    print("\n🏁 Sweep complete.")
    print(f"Best {monitor}: {best['score']:.6f}")
    print(f"Checkpoint: {best['ckpt']}")
    print(f"Params: {json.dumps(best['params'])}")
    return {"best": best, "results": results}
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.monker_parser import load_range_file_cached
from ml.utils.config import load_model_config
from ml.config.types_hands import ALL_HANDS, HAND_TO_ID

SCENARIO_KEYS = ["stack_bb", "hero_pos", "opener_pos", "opener_action"]

def _range_map_to_vector(rng_map: Dict[str, float]) -> np.ndarray:
    """Map a dict like {'AA':1.0,'AKs':0.2,...} → dense 169 vector in ALL_HANDS order."""
    vec = np.zeros(len(ALL_HANDS), dtype="float32")
    for hand, p in rng_map.items():
        idx = HAND_TO_ID.get(hand)
        if idx is not None:
            vec[idx] = float(p)
    s = vec.sum()
    if s > 0:
        vec /= s
    else:
        # fallback (uniform) if a file was empty or unparseable
        vec[:] = 1.0 / len(ALL_HANDS)
    return vec


def _aggregate_group(group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate all files in a scenario group into a single averaged, normalized range.
    group_df has columns SCENARIO_KEYS + ['abs_path'] (from monker_manifest).
    """
    # Average across files, then normalize
    acc: Dict[str, float] = {}
    n_files = 0
    for _, row in group_df.iterrows():
        rng_map = load_range_file_cached(Path(row["abs_path"]))
        if not rng_map:
            continue
        for k, v in rng_map.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n_files += 1

    if n_files == 0:
        # produce a uniform placeholder if nothing parsed
        vec = np.ones(len(ALL_HANDS), dtype="float32") / len(ALL_HANDS)
    else:
        # average
        for k in list(acc.keys()):
            acc[k] /= n_files
        vec = _range_map_to_vector(acc)

    # Emit row dict
    out: Dict[str, Any] = {k: group_df.iloc[0][k] for k in SCENARIO_KEYS}
    # y_0..y_168
    for i, p in enumerate(vec):
        out[f"y_{i}"] = float(p)
    out["weight"] = float(n_files)  # simple & effective: number of contributing files
    return out


def build_rangenet_preflop(manifest_path: Path, out_parquet: Path,
                           min_files: int | None = None) -> pd.DataFrame:
    """
    Read monker manifest → aggregate into one row per scenario with 169-dim soft label.
    """
    manifest = pd.read_parquet(manifest_path)
    need = set(SCENARIO_KEYS) | {"abs_path"}
    missing = sorted([c for c in need if c not in manifest.columns])
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Optional filter by minimum files per scenario (if the manifest already has counts)
    if min_files is not None and "n_files" in manifest.columns:
        manifest = manifest[manifest["n_files"] >= int(min_files)].reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for keys, g in manifest.groupby(SCENARIO_KEYS, as_index=False):
        row = _aggregate_group(g)
        rows.append(row)

    out_df = pd.DataFrame(rows)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"✅ wrote {out_parquet} with {len(out_df):,} scenarios")
    print("   Schema:",
          ", ".join(list(out_df.columns[:len(SCENARIO_KEYS)+3])) + ", … y_0..y_168 …, weight")
    return out_df


def run_from_config(cfg: Dict[str, Any]) -> None:
    """
    Expected YAML keys:
      inputs:
        monker_manifest: data/artifacts/monker_manifest.parquet
      outputs:
        rangenet_preflop: data/datasets/rangenet_preflop.parquet
      etl:
        min_files_per_scenario: 1
    """
    def get(path: str, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    manifest = Path(get("inputs.manifest"))
    out_path = Path(get("outputs.rangenet_preflop", "data/datasets/rangenet_preflop.parquet"))
    min_files = get("etl.min_files_per_scenario", None)

    if not manifest or not manifest.exists():
        raise FileNotFoundError(f"monker_manifest not found: {manifest}")

    build_rangenet_preflop(manifest, out_path, min_files=min_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rangenet/preflop",
                    help="Model name or YAML path (resolved by load_model_config)")
    # one-off override for output path
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.out:
        cfg.setdefault("outputs", {})["rangenet_preflop"] = args.out

    run_from_config(cfg)


if __name__ == "__main__":
    main()
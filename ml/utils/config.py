from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Mapping, Any
from dotenv import load_dotenv
from omegaconf import OmegaConf

CONFIG_ROOT = Path("ml/config")
load_dotenv()

# Optional: load .env once somewhere central in your app startup
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

CONFIG_ROOT = Path("ml/config")  # keep your existing value

def _resolve_paths(
    model: Optional[str],
    variant: Optional[str],
    profile: Optional[str],
    path: Optional[str],
) -> tuple[Path, Optional[Path]]:
    """
    Resolution order:
      1) If `path` points to a concrete YAML, use it directly (no merging).
      2) Else, resolve (model[/variant]/<profile>.yaml), where profile defaults to:
           profile arg → $ML_PROFILE → "base".
         If profile != "base", also load base.yaml and return (base, overlay).
    """
    # 1) explicit file path
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p, None

    if not model:
        raise ValueError("Either `path` or `model` must be provided.")

    # profile precedence: arg → env → "base"
    prof = (profile or os.getenv("ML_PROFILE") or "base").strip()

    # model / [variant] / profile.yaml
    base_dir = CONFIG_ROOT / model / variant if variant else CONFIG_ROOT / model
    cfg_path = base_dir / f"{prof}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path} "
            f"(model='{model}'{f', variant={variant}' if variant else ''}, profile='{prof}')"
        )

    # If profile != base, merge base <- profile (return both paths)
    if prof != "base":
        base_path = base_dir / "base.yaml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config required for merging but missing: {base_path}"
            )
        return base_path, cfg_path

    # profile == base → single file
    return cfg_path, None


def load_model_config(
    model: Optional[str] = None,
    *,
    variant: Optional[str] = None,
    profile: Optional[str] = None,
    path: Optional[str] = None,
) -> dict:
    """
    Load config by (model[/variant], profile) or direct path.
      - model="rangenet", variant="postflop", profile="dev.yaml"
      - model="populationnet", profile="prod"
      - path="ml/config/equitynet/postflop/prod.yaml"
    Returns a plain dict.
    """
    base_path, overlay_path = _resolve_paths(model, variant, profile, path)

    base_cfg = OmegaConf.load(str(base_path))
    if overlay_path:
        overlay_cfg = OmegaConf.load(str(overlay_path))
        merged = OmegaConf.merge(base_cfg, overlay_cfg)
    else:
        merged = base_cfg
    return OmegaConf.to_container(merged, resolve=True)  # -> dict


def dump_config_snapshot(cfg: Mapping[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), f=str(out_path))
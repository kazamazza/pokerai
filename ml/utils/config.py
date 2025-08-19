import copy
import os
from pathlib import Path
from typing import Dict, Any
import dotenv

dotenv.load_dotenv()

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_cfg(yaml_path="ml/config/settings.yaml"):
    import os, yaml
    cfg_all = yaml.safe_load(Path(yaml_path).read_text())

    tr_root = (cfg_all.get("training", {}) or {}).get("equitynet", {}) or {}
    profile = os.getenv("ML_PROFILE", tr_root.get("default_profile", "dev"))
    prof = (tr_root.get("profiles", {}) or {}).get(profile, {})
    if not prof:
        raise ValueError(f"training.equitynet profile '{profile}' not found")

    # merge common paths + profile hypers
    tcfg = {
        **{k:v for k,v in tr_root.items() if k not in ("profiles","default_profile")},
        **prof
    }
    seed = int(cfg_all.get("seed", 42))
    return cfg_all, tcfg, seed, profile



def select_profile(section: Dict[str, Any], profile_env: str|None) -> Dict[str, Any]:
    """
    If section has 'profiles', overlay the selected profile onto the base.
    Selection order: ENV VAR > section['profile'] (optional) > 'dev' if PY_ENV=development else None.
    """
    base = {k:v for k,v in section.items() if k != "profiles"}
    profiles = section.get("profiles") or {}

    # profile selection
    py_env = (os.getenv("PY_ENV") or "").lower()     # "development" / "production"
    default_guess = "dev" if py_env.startswith("dev") else None
    chosen = os.getenv("PROFILE") or profile_env or default_guess

    if chosen and chosen in profiles:
        return deep_merge(base, profiles[chosen])
    return base

def load_section(cfg: Dict[str, Any], key: str, profile_env: str|None=None) -> Dict[str, Any]:
    return select_profile(cfg.get(key, {}) or {}, profile_env)
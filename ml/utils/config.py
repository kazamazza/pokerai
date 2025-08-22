from pathlib import Path
import yaml

CONFIG_DIR = Path("ml/config")

def load_model_config(model_name: str) -> dict:
    """
    Load a YAML config for the given model.
    Example: load_model_config("equitynet") -> ml/config/equitynet.yaml
    """
    path = CONFIG_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
# ml/config/__init__.py
from dataclasses import dataclass
from pathlib import Path
import json, yaml

DEFAULT_YAML = Path("ml/config/settings.yaml")
AUTO_AXES    = Path("ml/config/axes.auto.json")

@dataclass(frozen=True)
class Axes:
    stacks: list[int]
    positions: list[str]
    action_contexts: list[str]
    rake_tiers: list[str]
    ante_values: list[float]
    open_size_policies: list[str]
    multiway_contexts: list[str]
    population_types: list[str]
    exploit_settings: list[str]

def load_settings(path: Path = DEFAULT_YAML) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_auto_axes(path: Path = AUTO_AXES) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def build_axes() -> Axes:
    s = load_settings()
    a = load_auto_axes()
    # merge: data-discovered values take precedence if present
    stacks = sorted(set(a.get("stacks", []))) or s["defaults"]["stacks"]
    positions = a.get("positions", s["defaults"]["positions"])
    contexts  = a.get("action_contexts", s["defaults"]["action_contexts"])

    return Axes(
        stacks=stacks,
        positions=positions,
        action_contexts=contexts,
        rake_tiers=s["rake_tiers"],
        ante_values=s["ante_values"],
        open_size_policies=s["open_size_policies"],
        multiway_contexts=s["multiway_contexts"],
        population_types=s["population_types"],
        exploit_settings=s["exploit_settings"],
    )
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class StakeSolverParams:
    raise_mult: List[float]
    allin_gate_spr: float

def load_stake_params(solver_yaml: Dict[str, Any], stake_key: str) -> StakeSolverParams:
    stake = solver_yaml.get(stake_key)
    if not isinstance(stake, dict):
        raise ValueError(f"stake_key not found in solver_yaml: {stake_key}")

    raise_mult = stake.get("raise_mult")
    if not isinstance(raise_mult, list) or not raise_mult:
        raise ValueError(f"{stake_key}.raise_mult missing/invalid")

    allin_gate_spr = stake.get("allin_gate_spr")
    if allin_gate_spr is None:
        raise ValueError(f"{stake_key}.allin_gate_spr missing")

    return StakeSolverParams(
        raise_mult=[float(x) for x in raise_mult],
        allin_gate_spr=float(allin_gate_spr),
    )
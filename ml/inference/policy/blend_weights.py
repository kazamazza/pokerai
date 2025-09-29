from dataclasses import dataclass


@dataclass
class BlendWeights:
    """
    How to mix population baseline with exploit deltas.
    w_exploit: 0..1  — hard weight for exploit (override by 'weight' in exploit output if present)
    weight_cap: caps the per-spot exploit weight coming back from exploit model
    """
    w_exploit: float = 0.30
    weight_cap: float = 1.00
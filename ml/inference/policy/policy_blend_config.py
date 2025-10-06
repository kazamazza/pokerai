from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PolicyBlendConfig:
    """
    Master configuration for blending PostflopPolicy, Range/Equity,
    PopNet, and ExploitNet signals into a unified decision policy.

    This defines the relative weights, temperature, and safety guardrails
    used by the PolicyInfer engine.
    """

    # --- weighting of each adjustment component ---
    lambda_eq: float = 0.8       # equity influence (0–1)
    lambda_pop: float = 0.4      # population bias weight
    lambda_expl: float = 0.4     # exploit bias weight
    lambda_risk: float = 0.2     # risk aversion weight

    # --- temperature & smoothing ---
    temperature: float = 0.95    # sharpness of final softmax (1.0 = neutral)
    tie_mix_threshold: float = 0.03  # if top-2 probs within this, mix actions
    epsilon_explore: float = 0.02    # small random exploration mass

    # --- legality & sanity ---
    min_legal_prob: float = 1e-5     # ensures softmax never collapses
    enforce_legality: bool = True    # mask out impossible actions

    # --- risk guardrails ---
    max_allin_freq: float = 0.10     # cap for ALLIN in non-shortstack situations
    risk_floor_stack_bb: float = 20  # if eff_stack > this, reduce all-in freq

    # --- logging / diagnostics ---
    debug: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dict for logs or JSON export."""
        return {
            "lambda_eq": self.lambda_eq,
            "lambda_pop": self.lambda_pop,
            "lambda_expl": self.lambda_expl,
            "lambda_risk": self.lambda_risk,
            "temperature": self.temperature,
            "tie_mix_threshold": self.tie_mix_threshold,
            "epsilon_explore": self.epsilon_explore,
            "min_legal_prob": self.min_legal_prob,
            "enforce_legality": self.enforce_legality,
            "max_allin_freq": self.max_allin_freq,
            "risk_floor_stack_bb": self.risk_floor_stack_bb,
            "debug": self.debug,
            "extra": self.extra,
        }

    @classmethod
    def default(cls) -> "PolicyBlendConfig":
        """Return a safe, balanced default configuration."""
        return cls()
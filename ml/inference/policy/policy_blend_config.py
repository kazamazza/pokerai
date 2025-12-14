from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict

@dataclass
class PolicyBlendConfig:
    # ===== core (existing) =====
    temperature: float = 1.0
    min_legal_prob: float = 1e-6
    tie_mix_threshold: float = 0.02
    epsilon_explore: float = 0.00
    lambda_eq: float = 0.0
    eq_min_abs_margin: float = 0.01
    eq_max_logit_delta: float = 2.0
    lambda_expl: float = 0.0
    risk_floor_stack_bb: float = 0.0
    max_allin_freq: float = 0.20
    equity_nudge_pre: float = 0.02  # preflop only

    # ===== legacy single-bucket bias (kept for back-compat; tuner can ignore) =====
    single_bucket_raise_bias: float = 2.5
    raise_bias_min_size: float = 0.30
    raise_bias_min_eq: float = 0.00
    raise_bias_eq_boost_gate: float = 0.62
    raise_bias_eq_boost: float = 1.5
    # Safety limits & context guards
    raise_max_logit_boost: float = 8.0
    raise_when_faced_min_size: float = 0.30   # require faced size ≥ this to bias raises
    raise_when_faced_max_size: float = 1.00   # optional upper bound; leave at 1.0 to disable
    raise_block_if_allin_legal: bool = False   # don't push raises if ALLIN is legal (optional)

    # ===== NEW: global tuner toggles =====
    enable_tuner: bool = True
    tuner_debug: bool = True  # surface tuner info in PolicyResponse.debug
    tuner_step: float = 0.50  # move only a fraction toward target tau (0..1)
    tau_floor: float = 0.02   # lower/upper rails for target tau
    tau_ceil: float = 0.60

    # ===== NEW: Facing — signal-driven raise share =====
    # Base/strong targets for the *sum* of legal RAISE_* (often just RAISE_200 at NL10)
    raise_min_share_tau: float = 0.12
    raise_tau_when_strong: float = 0.25
    raise_tau_equity_gate: float = 0.62  # p_win threshold for "strong"

    # Equity → tau nudge (smooth, bounded)
    eq_tau_gate: float = 0.56            # start adding raise when p_win > this
    eq_tau_scale: float = 0.80           # tau gain per unit (p_win - gate)
    eq_tau_max: float = 0.18             # cap on equity-based tau add

    # Exploit (prob-style) → tau nudges (smooth, bounded)
    # Gates interpret advantage over the best alternative bucket.
    expl_fold_gate: float = 0.10         # fold advantage gate to start nudging
    expl_fold_scale: float = 0.50        # tau gain per unit fold advantage
    expl_fold_max: float = 0.20

    expl_aggr_gate: float = 0.10         # raise (villain-aggressive) advantage gate
    expl_aggr_scale: float = 0.25
    expl_aggr_max: float = 0.08



    # ===== Root — target-share for bets (sum over BET_* [+ DONK_33 if OOP]) =====
    bet_min_share_tau: float = 0.25
    bet_tau_when_strong: float = 0.45
    bet_tau_equity_gate: float = 0.60
    bet_tau_expl_fold_boost: float = 0.10
    bet_max_logit_boost: float = 8.0

    @classmethod
    def default(cls) -> "PolicyBlendConfig":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyBlendConfig":
        if not isinstance(data, dict):
            return cls.default()
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        cfg = cls(**kwargs)
        return cfg.validate()

    def updated(self, **overrides: Any) -> "PolicyBlendConfig":
        base = self.to_dict()
        base.update({k: v for k, v in overrides.items() if k in base})
        return PolicyBlendConfig.from_dict(base)

    def validate(self) -> "PolicyBlendConfig":
        # core clamps
        self.temperature = float(max(self.temperature, 1e-6))
        self.min_legal_prob = float(max(self.min_legal_prob, 0.0))
        self.tie_mix_threshold = float(max(self.tie_mix_threshold, 0.0))
        self.epsilon_explore = float(min(max(self.epsilon_explore, 0.0), 1.0))
        self.lambda_eq = float(max(self.lambda_eq, 0.0))
        self.eq_min_abs_margin = float(max(self.eq_min_abs_margin, 0.0))
        self.eq_max_logit_delta = float(max(self.eq_max_logit_delta, 0.0))
        self.lambda_expl = float(max(self.lambda_expl, 0.0))
        self.risk_floor_stack_bb = float(max(self.risk_floor_stack_bb, 0.0))
        self.max_allin_freq = float(min(max(self.max_allin_freq, 0.0), 1.0))
        self.equity_nudge_pre = float(self.equity_nudge_pre)

        # legacy bias clamps
        self.single_bucket_raise_bias = float(max(self.single_bucket_raise_bias, 0.0))
        self.raise_bias_min_size = float(min(max(self.raise_bias_min_size, 0.0), 1.0))
        self.raise_bias_min_eq = float(max(self.raise_bias_min_eq, 0.0))
        self.raise_bias_eq_boost_gate = float(max(self.raise_bias_eq_boost_gate, 0.0))
        self.raise_bias_eq_boost = float(max(self.raise_bias_eq_boost, 0.0))

        # tuner clamps/rails
        def _ct(x: float, lo=0.0, hi=0.9) -> float:
            return float(min(max(x, lo), hi))

        self.tuner_step = float(min(max(self.tuner_step, 0.0), 1.0))
        self.tau_floor = _ct(self.tau_floor, 0.0, 0.95)
        self.tau_ceil = _ct(self.tau_ceil, self.tau_floor, 0.98)

        # facing raise taus
        self.raise_min_share_tau = _ct(self.raise_min_share_tau)
        self.raise_tau_when_strong = _ct(self.raise_tau_when_strong)
        self.raise_tau_equity_gate = float(max(self.raise_tau_equity_gate, 0.0))

        # equity → tau
        self.eq_tau_gate = float(max(self.eq_tau_gate, 0.0))
        self.eq_tau_scale = float(max(self.eq_tau_scale, 0.0))
        self.eq_tau_max = float(max(self.eq_tau_max, 0.0))

        # exploit → tau
        self.expl_fold_gate = float(max(self.expl_fold_gate, 0.0))
        self.expl_fold_scale = float(max(self.expl_fold_scale, 0.0))
        self.expl_fold_max = float(max(self.expl_fold_max, 0.0))

        self.expl_aggr_gate = float(max(self.expl_aggr_gate, 0.0))
        self.expl_aggr_scale = float(max(self.expl_aggr_scale, 0.0))
        self.expl_aggr_max = float(max(self.expl_aggr_max, 0.0))

        # safety & size gates
        self.raise_max_logit_boost = float(max(self.raise_max_logit_boost, 0.0))
        self.raise_when_faced_min_size = float(min(max(self.raise_when_faced_min_size, 0.0), 1.0))
        self.raise_when_faced_max_size = float(min(max(self.raise_when_faced_max_size, 0.0), 1.0))

        # root bet taus
        self.bet_min_share_tau = _ct(self.bet_min_share_tau)
        self.bet_tau_when_strong = _ct(self.bet_tau_when_strong)
        self.bet_tau_equity_gate = float(max(self.bet_tau_equity_gate, 0.0))
        self.bet_tau_expl_fold_boost = _ct(self.bet_tau_expl_fold_boost)
        self.bet_max_logit_boost = float(max(self.bet_max_logit_boost, 0.0))

        return self
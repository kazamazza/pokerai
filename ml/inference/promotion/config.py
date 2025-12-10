from dataclasses import dataclass


@dataclass
class PromotionConfig:
    # weights for score ∈ [0, 1]
    w_ev: float = 0.60
    w_eq: float = 0.30
    w_expl: float = 0.10

    # gates & scales
    eq_gate: float = 0.55          # need at least this equity before promotion grows
    expl_gate: float = 0.05        # opponent bias needs to exceed this
    ev_cap_bb: float = 3.0         # cap positive EV gap for normalization (bb)

    # map score to target share
    tau_min: float = 0.12
    tau_max: float = 0.35

    # safety rails
    max_logit_boost: float = 8.0
    respect_fold_when_facing: bool = True  # do not override pure GTO FOLD baseline by default
    cap_allin: bool = True

    # optional: temperature scaling after promotion (gentle)
    min_temp: float = 0.6
    max_temp: float = 1.2
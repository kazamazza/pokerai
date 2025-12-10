from dataclasses import dataclass


@dataclass
class PromotionConfig:
    # weights & gates
    w_ev: float = 0.60
    w_eq: float = 0.30
    w_expl: float = 0.10
    ev_cap_bb: float = 3.0
    eq_gate: float = 0.55
    expl_gate: float = 0.05

    # target share mapping
    tau_min: float = 0.12
    tau_max: float = 0.35
    max_logit_boost: float = 8.0

    # temp rails
    min_temp: float = 0.6
    max_temp: float = 1.2

    # policy rails
    respect_fold_when_facing: bool = True
    cap_allin: bool = True

    # strong-hand guard
    strong_eq_floor: float = 0.80
    strong_ev_margin_bb: float = 0.50
    fold_cap_when_strong: float = 0.005

    # EV scale sanity (bb vs centi-bb)
    ev_is_centi_bb: bool = False

    @staticmethod
    def default() -> "PromotionConfig":
        return PromotionConfig()

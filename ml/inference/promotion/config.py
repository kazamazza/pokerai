from dataclasses import dataclass

@dataclass
class PromotionConfig:
    w_ev: float = 0.50   # was 0.60
    w_eq: float = 0.35   # was 0.30
    w_expl: float = 0.15 # was 0.10
    ev_cap_bb: float = 4.0   # was 3.0
    eq_gate: float = 0.50    # was 0.55
    expl_gate: float = 0.02  # was 0.05
    tau_min: float = 0.12
    tau_max: float = 0.45    # was 0.35
    max_logit_boost: float = 10.0  # was 8.0
    min_temp: float = 0.6
    max_temp: float = 1.2
    respect_fold_when_facing: bool = True
    cap_allin: bool = True
    strong_eq_floor: float = 0.80
    strong_ev_margin_bb: float = 0.50
    fold_cap_when_strong: float = 0.005
    ev_is_centi_bb: bool = False
    min_score_for_promo: float = 0.05
    facing_ev_gap_override_bb: float = 1.0  # promo if best_raise - call ≥ this
    facing_override_share: float = 0.22  # fixed share when override trips

    @staticmethod
    def default() -> "PromotionConfig":
        return PromotionConfig()

from ml.inference.policy.policy_blend_config import PolicyBlendConfig

PROFILES = {
    "root_bet_influenced": {
        "enable_tuner": True,
        "tuner_debug": True,
        "temperature": 1.0,
        "lambda_expl": 2.2,
        "lambda_eq": 0.5,

        "step": 0.7,
        "tau_floor": 0.05,
        "tau_ceil": 0.65,

        "eq_tau_gate": 0.48,
        "eq_tau_scale": 1.6,
        "eq_tau_max": 0.25,

        "expl_fold_gate": 0.00,
        "expl_fold_scale": 3.5,
        "expl_fold_max": 0.45,

        "expl_aggr_gate": 0.00,
        "expl_aggr_scale": 1.2,
        "expl_aggr_max": 0.20,

        "raise_max_logit_boost": 10.0,
        "min_gto_engage_prob": 0.01,
    },

    "facing_raise_influenced": {
        "enable_tuner": True,
        "tuner_debug": True,
        "temperature": 1.0,
        "lambda_expl": 2.5,
        "lambda_eq": 0.5,

        "step": 0.8,
        "tau_floor": 0.08,
        "tau_ceil": 0.80,

        "eq_tau_gate": 0.50,
        "eq_tau_scale": 2.2,
        "eq_tau_max": 0.35,

        "raise_tau_equity_gate": 0.55,
        "raise_tau_when_strong": 0.55,
        "raise_min_share_tau": 0.20,

        "expl_fold_gate": 0.00,
        "expl_fold_scale": 4.0,
        "expl_fold_max": 0.50,

        "expl_aggr_gate": 0.00,
        "expl_aggr_scale": 1.4,
        "expl_aggr_max": 0.30,

        "raise_when_faced_min_size": 0.30,
        "raise_when_faced_max_size": 1.00,
        "raise_block_if_allin_legal": True,
        "raise_max_logit_boost": 12.0,

        "min_gto_engage_prob": 0.01,
    },
}

def make_blend(name: str) -> PolicyBlendConfig:
    """Build a validated PolicyBlendConfig from a profile name."""
    return PolicyBlendConfig.from_dict(PROFILES[name])
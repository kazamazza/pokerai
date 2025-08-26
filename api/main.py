# api/main.py
from fastapi import FastAPI
from api.schemas import PolicyRequest
from ml.inference.equity import EquityNetInfer
from ml.inference.policy import PolicyInfer
from ml.inference.rangenet import RangeNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInfer
from ml.features.boards import load_board_clusterer


def build_policy_infer() -> PolicyInfer:
    RANGE_CKPT = "checkpoints/rangenet_postflop.ckpt"
    RANGE_SIDECAR = "checkpoints/rangenet_postflop_sidecar.json"

    EXPLOIT_CKPT = "checkpoints/exploitnet.ckpt"
    EXPLOIT_SIDECAR = "checkpoints/exploitnet_sidecar.json"

    POP_CKPT = "checkpoints/populationnet.ckpt"
    POP_SIDECAR = "checkpoints/populationnet_sidecar.json"

    EQ_PRE_CKPT = "checkpoints/equity_pre.ckpt"
    EQ_PRE_SIDECAR = "checkpoints/equity_pre_sidecar.json"

    EQ_POST_CKPT = "checkpoints/equity_post.ckpt"
    EQ_POST_SIDECAR = "checkpoints/equity_post_sidecar.json"

    # Instantiation
    rng_infer = RangeNetInfer.from_checkpoint(RANGE_CKPT, RANGE_SIDECAR)
    expl_infer = ExploitNetInfer.from_checkpoint(EXPLOIT_CKPT, EXPLOIT_SIDECAR)
    pop_infer = PopulationNetInfer.from_checkpoint(POP_CKPT, POP_SIDECAR)
    eq_pre = EquityNetInfer.from_checkpoint(EQ_PRE_CKPT, EQ_PRE_SIDECAR)
    clusterer = load_board_clusterer({
        "board_clustering": {
            "type": "rule",     # or "kmeans" depending on what you trained
            "artifact": None,
            "n_clusters": 48,
        }
    })

    params = {
        "alpha": 0.35,
        "beta": 0.35,
        "gamma": 0.30,
        "exploit_full_weight": 200,
        "spr_shove_threshold": 1.5,
    }

    return PolicyInfer(
        pop_infer=pop_infer,
        exploit_infer=expl_infer,
        range_infer=rng_infer,
        equity_infer_pre=eq_pre,
        board_clusterer=clusterer,
        params=params,
    )


app = FastAPI(title="Poker Policy API")
policy_infer = build_policy_infer()


# ---- routes ----
@app.post("/policy/act")
def policy_act(req: PolicyRequest):
    d = req.to_infer_dict()
    out = policy_infer.predict(d)
    return out
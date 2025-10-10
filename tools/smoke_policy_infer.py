import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.inference.equity import EquityNetInfer
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer
from ml.inference.population import PopulationNetInference
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import PreflopPolicy
from ml.inference.player_exploit_store import PlayerExploitStore, ExploitConfig
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.features.boards import load_board_clusterer

# === Hard-coded paths (edit if different) ===
POSTFLOP_DIR = Path("checkpoints/postflop_policy")
PREFLOP_DIR  = Path("checkpoints/range_pre")
EQUITY_DIR   = Path("checkpoints/equitynet")
POP_DIR      = Path("checkpoints/popnet")
CLUSTER_JSON = Path("data/artifacts/board_clusters_kmeans_128.json")

# --- tiny helper to build a PolicyRequest dict matching your dataclass ---
def make_request(
    *,
    street: int = 0,
    ip_pos: str | None = None,
    oop_pos: str | None = None,
    hero_pos: str | None = None,
    villain_pos: str | None = None,
    hero_hand: str | None = None,
    board: str | None = None,
    pot_bb: float = 0.0,
    eff_stack_bb: float = 100.0,
    facing_bet: bool = False,
    villain_id: str | None = None,
    actions_hist: list | None = None,
    raw: dict | None = None,
) -> dict:
    """Return a dict that can be passed to PolicyInfer.predict(...)."""
    return {
        "street": int(street or 0),
        "ip_pos": ip_pos,
        "oop_pos": oop_pos,
        "hero_pos": hero_pos,
        "villain_pos": villain_pos,
        "hero_hand": hero_hand,
        "board": board,
        "pot_bb": float(pot_bb or 0.0),
        "eff_stack_bb": float(eff_stack_bb or 100.0),
        "facing_bet": bool(facing_bet),
        "villain_id": villain_id,
        "actions_hist": list(actions_hist) if actions_hist else None,
        "raw": dict(raw) if raw else {},
    }


def load_dependency(cls, arg):
    """Small loader that prints warnings on failure and returns None on error."""
    if cls is None:
        print(f"  ⚠️  NOT AVAILABLE: {arg} loader missing from imports.")
        return None
    try:
        return cls.from_dir(str(arg)) if hasattr(cls, "from_dir") else cls.load(str(arg))
    except Exception as e:
        print(f"  ⚠️  Failed to load {arg}: {e}")
        return None


def main():
    print("🚀 PolicyInfer minimal smoke test starting...")

    # 1) load components (best-effort)
    print("• loading components (best-effort):")
    postflop = load_dependency(PostflopPolicyInfer, POSTFLOP_DIR)
    preflop  = load_dependency(PreflopPolicy, PREFLOP_DIR)
    equity   = load_dependency(EquityNetInfer, EQUITY_DIR)
    pop      = load_dependency(PopulationNetInference, POP_DIR)
    expl = PlayerExploitStore(ExploitConfig())

    cfg = {
        "board_clustering": {
            "type": "kmeans",
            "artifact": "data/artifacts/board_clusters_kmeans_128.json"
        }
    }

    clusterer = load_board_clusterer(cfg)
    print("✅ Loaded KMeans clusterer:", type(clusterer))
    print("Example cluster id:", clusterer.predict_one("AsKh2d"))

    # 2) build deps object (PolicyInfer expects some of these; ensure required ones exist)
    deps = PolicyInferDeps(
        pop=pop,
        exploit=expl,            # you can inject PlayerExploitStore here if available
        equity=equity,
        range_pre=preflop,
        policy_post=postflop,
        clusterer=clusterer,
        params={},
    )

    # 3) make PolicyInfer instance (will raise if mandatory deps missing in its ctor)
    try:
        infer = PolicyInfer(deps, blend_cfg=PolicyBlendConfig())
    except Exception as e:
        print("❌ Failed to construct PolicyInfer:", e)
        print("   Make sure mandatory infer components (postflop/preflop/equity/exploit) exist.")
        return

    # 4) sample requests: a few deterministic + a simple random stress generator
    sample_requests = [
        make_request(street=0, ip_pos="BTN", oop_pos="BB", hero_hand="AsAh"),
        make_request(street=1, ip_pos="BTN", oop_pos="BB", board="AsKh2d", pot_bb=18.0, eff_stack_bb=100.0),
        make_request(street=2, ip_pos="CO", oop_pos="BTN", board="AsKh2d7s", pot_bb=30.0, eff_stack_bb=60.0, facing_bet=True),
    ]

    # Add N random test requests (lightweight) — change n to stress-test
    n_random = 5
    POS = ["UTG","HJ","CO","BTN","SB","BB","IP","OOP"]
    RANKS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
    SUITS = ["s","h","d","c"]

    def random_board(n_cards):
        cards = []
        seen = set()
        while len(cards) < n_cards:
            c = random.choice(RANKS) + random.choice(SUITS)
            if c in seen: continue
            seen.add(c)
            cards.append(c)
        return "".join(cards)

    for i in range(n_random):
        st = random.choice([1,2,3])
        board = random_board(3 if st==1 else (4 if st==2 else 5))
        sample_requests.append(
            make_request(
                street=st,
                ip_pos=random.choice(POS),
                oop_pos=random.choice(POS),
                board=board,
                pot_bb=random.choice([5.0, 10.0, 20.0, 50.0]),
                eff_stack_bb=random.choice([50.0, 100.0, 200.0]),
                facing_bet=random.choice([False, True]),
                villain_id=random.choice([None, "player_123", "v_abc"]),
            )
        )

    # 5) run requests and print brief output
    print("\n• running requests:")
    for i, req in enumerate(sample_requests, 1):
        try:
            start = time.time()
            out = infer.predict(req)
            dt = time.time() - start
            top_idx = int(max(range(len(out.probs)), key=lambda j: out.probs[j])) if out.probs else -1
            top_action = out.actions[top_idx] if (top_idx >= 0 and out.actions) else "NONE"
            print(f"[{i:02d}] street={req['street']:d} board={req.get('board','')} top={top_action:8} p={out.probs[top_idx]:.3f}  dt={dt*1000:.0f}ms")
            # Print debug snippet
            dbg = out.debug or {}
            snippet = {k: dbg.get(k) for k in ("actor","ctx","street","board_cluster_id","exploit_debug") if k in dbg}
            if snippet:
                print("     debug:", snippet)
        except Exception as e:
            print(f"[{i:02d}] ERROR running infer.predict: {e}")

    # 6) optional stress loop (uncomment if you want to try many)
    # n_iter = 100000
    # for idx in range(n_iter):
    #     r = random.choice(sample_requests)
    #     _ = infer.predict(r)  # ignore outputs, just stress

    print("\n✅ Smoke test finished.")


if __name__ == "__main__":
    main()
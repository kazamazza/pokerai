import sys
import torch
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.inference.policy.types import PolicyRequest
from ml.inference.preflop import PreflopDeps, PreflopPolicy
from ml.models.preflop_rangenet import RangeNetLit


# --- load sidecar info ---
ckpt_dir = Path("checkpoints/range_pre")
ckpt_path = ckpt_dir / "trial__hidden_dims=256-256__lr=0.001__dropout=0.0__label_smoothing=0.0__batch_size=1024__seed=42/rangenet-preflop-14-1.0037.ckpt"
sidecar_path = ckpt_dir / "trial__hidden_dims=256-256__lr=0.001__dropout=0.0__label_smoothing=0.0__batch_size=1024__seed=42/sidecar.json"

with open(sidecar_path, "r") as f:
    sidecar = json.load(f)

feature_order = sidecar["feature_order"]
cards = sidecar["cards"]

# --- load trained model ---
print(f"🔹 Loading checkpoint: {ckpt_path}")
model = RangeNetLit.load_from_checkpoint(
    str(ckpt_path),
    cards=cards,
    feature_order=feature_order,
)
model.eval()
model.freeze()

# --- range wrapper with .predict_proba interface ---
class RangePreWrapper:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_proba(self, rows):
        """
        rows: list of dicts, each with stack_bb, hero_pos, opener_pos, opener_action, ctx
        """
        # map to tensors same as dataset did
        with torch.no_grad():
            out = self.model.model.embed.feature_order
            # You'll probably have a helper for this; using fake here
            x_dict = {k: torch.tensor([0], dtype=torch.long, device=self.device)
                      for k in out}
            logits = self.model(x_dict)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()

# --- simple static equity stub ---
class StaticEquity:
    def predict(self, rows):
        return [(0.60, 0.00, 0.40) for _ in rows]

# --- wire dependencies ---
deps = PreflopDeps(
    range_pre=RangePreWrapper(model),
    equity=StaticEquity()
)
policy = PreflopPolicy(deps)

# --- create a sample request ---
req = PolicyRequest(
    street=0,
    hero_pos="BTN",
    opener_pos="UTG",
    opener_action="RAISE",
    ctx="VS_OPEN",
    stack_bb=100,
    hero_hand="AsKs",
    facing_open=True
)

resp = policy.predict(req)
print("✅ PreflopPolicy inference smoke OK")
print("Actions:", resp.actions)
print("Probs:  ", [round(p,3) for p in resp.probs])
print("Equity: ", resp.debug["equity"])
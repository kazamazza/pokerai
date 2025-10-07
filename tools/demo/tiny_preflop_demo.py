# tiny_preflop_demo.py
import json, random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 0) Problem definition
# ---------------------------
ACTION_VOCAB = ["H1","H2","H3","H4","H5"]  # pretend "hand buckets"
HAND_COUNT   = len(ACTION_VOCAB)

HERO_POS   = ["BTN","SB","BB"]
OPENER_POS = ["UTG","CO","BTN"]
CTX        = ["SRP","VS_3BET"]
OPEN_ACT   = ["RAISE","LIMP"]

FEATURE_ORDER = ["hero_pos", "opener_pos", "ctx", "opener_action"]  # categorical only here
CONT_FEATURES: List[str] = []  # none in this minimal demo; easy to add later

# ---------------------------
# 1) Make synthetic training data (with soft labels)
# ---------------------------
def _rule_logit(hero: str, opener: str, ctx: str, act: str) -> torch.Tensor:
    """
    Hand-crafted 'truth'. Returns 5 logits (one per action bucket).
    You can read & understand exactly why each bucket goes up/down.
    """
    base = torch.zeros(HAND_COUNT)

    # BTN advantage for H1/H2
    if hero == "BTN":
        base += torch.tensor([2.0, 1.0, 0.0, 0.0, -0.5])

    # VS_3BET shifts mass toward H3
    if ctx == "VS_3BET":
        base += torch.tensor([0.0, 0.5, 1.8, 0.0, 0.0])

    # If opener is UTG and act=RAISE, nudge towards tighter buckets (H1)
    if opener == "UTG" and act == "RAISE":
        base += torch.tensor([0.6, 0.2, 0.0, -0.2, -0.2])

    # If opener is BTN and act=LIMP, nudge to looser (H4/H5)
    if opener == "BTN" and act == "LIMP":
        base += torch.tensor([ -0.2, 0.0, 0.0, 0.7, 0.5 ])

    # tiny noise to avoid perfect determinism
    base += 0.1 * torch.randn(HAND_COUNT)
    return base

def synth_row() -> Tuple[Dict[str,str], np.ndarray]:
    hero   = random.choice(HERO_POS)
    opener = random.choice(OPENER_POS)
    ctx    = random.choice(CTX)
    act    = random.choice(OPEN_ACT)
    logits = _rule_logit(hero, opener, ctx, act)
    y_soft = F.softmax(logits, dim=0).numpy()  # soft label (sums to 1)
    return {"hero_pos": hero, "opener_pos": opener, "ctx": ctx, "opener_action": act}, y_soft

def make_dataset(n=2000, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X: List[Dict[str,str]] = []
    Y: List[np.ndarray] = []
    for _ in range(n):
        xd, y = synth_row()
        X.append(xd); Y.append(y)
    W = np.ones(n, dtype=np.float32)  # uniform weights
    return X, np.stack(Y).astype("float32"), W

# ---------------------------
# 2) Build encoders & id_maps for categoricals (cards)
# ---------------------------
def build_id_maps(
    X: List[Dict[str,str]], feature_order: Sequence[str]
) -> Tuple[Dict[str,Dict[str,int]], Dict[str,int]]:
    id_maps: Dict[str, Dict[str,int]] = {}
    cards: Dict[str, int] = {}
    for feat in feature_order:
        vals = sorted({row[feat] for row in X})
        enc = {v:i for i,v in enumerate(vals)}
        id_maps[feat] = enc
        cards[feat]   = len(enc)
    return id_maps, cards

def encode_X(X: List[Dict[str,str]], feature_order: Sequence[str], id_maps: Dict[str,Dict[str,int]]) -> Dict[str, torch.Tensor]:
    enc: Dict[str, List[int]] = {f: [] for f in feature_order}
    for row in X:
        for f in feature_order:
            enc[f].append(id_maps[f][row[f]])
    return {k: torch.tensor(v, dtype=torch.long) for k,v in enc.items()}

# ---------------------------
# 3) Model: embeddings -> MLP -> logits[5]
# ---------------------------
class CatEmbedBlock(nn.Module):
    def __init__(self, cards: Dict[str,int], feature_order: Sequence[str], max_emb_dim: int = 16):
        super().__init__()
        self.feature_order = list(feature_order)
        self.embs = nn.ModuleDict()
        for name in self.feature_order:
            c = int(cards[name])
            # tiny rule: embed dim grows with cardinality up to a cap
            d = min(max_emb_dim, max(4, int(round((c**0.5)*4))))
            self.embs[name] = nn.Embedding(c, d)
        self.out_dim = sum(e.embedding_dim for e in self.embs.values())

    def forward(self, x_cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = [ self.embs[name](x_cat[name]) for name in self.feature_order ]
        return torch.cat(outs, dim=-1)  # [B, sum(emb_dims)]

class ToyRangeNet(nn.Module):
    def __init__(self, cards: Dict[str,int], feature_order: Sequence[str], hidden=(64,64), dropout=0.1):
        super().__init__()
        self.embed = CatEmbedBlock(cards, feature_order)
        layers: List[nn.Module] = []
        in_dim = self.embed.out_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp  = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, HAND_COUNT)

    def forward(self, x_cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.embed(x_cat)      # [B, E]
        h = self.mlp(z)            # [B, H]
        return self.head(h)        # [B, 5] logits

# ---------------------------
# 4) Loss = KL( y_true || softmax(logits) )
# ---------------------------
def kl_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    p_log = F.log_softmax(logits, dim=-1)
    # KL(y||p) = sum y*(log y - log p)
    kl = torch.sum(y * (torch.log(y + 1e-8) - p_log), dim=-1)  # [B]
    return torch.sum(w * kl) / (w.sum() + 1e-8)

# ---------------------------
# 5) Train
# ---------------------------
def train_demo():
    X, Y, W = make_dataset(n=3000, seed=42)
    id_maps, cards = build_id_maps(X, FEATURE_ORDER)
    x_cat = encode_X(X, FEATURE_ORDER, id_maps)  # each key → LongTensor[B]
    y = torch.tensor(Y)                           # [B,5], sums≈1
    w = torch.tensor(W)                           # [B]

    model = ToyRangeNet(cards, FEATURE_ORDER, hidden=(64,64), dropout=0.10)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    B = y.shape[0]
    batch_size = 256
    epochs = 10

    for ep in range(1, epochs+1):
        # simple manual batching
        perm = torch.randperm(B)
        total = 0.0
        model.train()
        for i in range(0, B, batch_size):
            idx = perm[i:i+batch_size]
            xb = {k: v[idx] for k,v in x_cat.items()}
            yb = y[idx]
            wb = w[idx]
            logits = model(xb)
            loss = kl_loss(logits, yb, wb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        print(f"epoch {ep:02d}  train_kl={total/B:.4f}")

    # quick evaluation on a few hand-picked scenarios:
    model.eval()
    def predict(hero, opener, ctx, act):
        row = [{"hero_pos":hero,"opener_pos":opener,"ctx":ctx,"opener_action":act}]
        enc = encode_X(row, FEATURE_ORDER, id_maps)
        with torch.no_grad():
            logits = model(enc)
            probs = torch.softmax(logits, dim=-1)[0].tolist()
        top = sorted(zip(ACTION_VOCAB, probs), key=lambda x:x[1], reverse=True)
        return top, probs

    print("\n--- sanity predictions ---")
    for case in [
        ("BTN","UTG","SRP","RAISE"),
        ("BTN","UTG","VS_3BET","RAISE"),
        ("SB","BTN","VS_3BET","LIMP"),
        ("BB","BTN","SRP","LIMP"),
    ]:
        top, probs = predict(*case)
        pretty = ", ".join(f"{a}={p:.3f}" for a,p in top[:3])
        print(f"{case} → top3: {pretty}")

    # ---------------------------
    # 6) Sidecar (for inference later)
    # ---------------------------
    sidecar = {
        "sidecar_version": 1,
        "model_name": "ToyRangeNet",
        "feature_order": FEATURE_ORDER,        # ← order matters at inference
        "cards": cards,                        # ← vocab sizes
        "id_maps": id_maps,                    # ← string→id mapping used at training
        "cont_features": CONT_FEATURES,        # ← none here
        "action_vocab": ACTION_VOCAB,          # ← label order (5 buckets)
        "notes": "Demo sidecar for a tiny categorical-only range predictor."
    }
    print("\n--- sidecar ---")
    print(json.dumps(sidecar, indent=2))

if __name__ == "__main__":
    train_demo()
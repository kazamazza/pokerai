from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from ml.models.population_net import PopulationNetLit
from ml.utils.sidecar import load_sidecar


ACTIONS = ["FOLD", "CALL", "RAISE"]  # matches p_fold/p_call/p_raise order

class PopulationNetInference:
    """
    Inference for PopulationNetLit checkpoints.
    - Uses sidecar for feature_order + cards.
    - Reads embedding dims from checkpoint tensors to avoid any heuristic mismatch.
    - Accepts integer IDs for all categorical features.
    """

    def __init__(self, ckpt_path: str | Path, device: Optional[torch.device] = None):
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        # Sidecar sits next to the ckpt and uses the ".sidecar.json" suffix
        sidecar_path = self.ckpt_path.with_suffix(self.ckpt_path.suffix + ".sidecar.json")
        sc = load_sidecar(sidecar_path)
        self.feature_order: List[str] = list(sc["feature_order"])
        self.cards: Dict[str, int] = {str(k): int(v) for k, v in sc["cards"].items()}

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load checkpoint object
        obj = torch.load(self.ckpt_path, map_location=self.device)
        if not isinstance(obj, dict):
            raise RuntimeError("Unexpected checkpoint format; expected a dict-like object.")

        # Prefer Lightning's "state_dict"
        state: Dict[str, torch.Tensor]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state = obj["model_state_dict"]
        else:
            # sometimes the whole dict is just a state dict
            state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if not state:
                raise RuntimeError("Could not find a state_dict in checkpoint.")

        # Extract exact embedding dims from tensors: emb_layers.<feat>.weight [rows, dim]
        emb_dims: Dict[str, int] = {}
        rows_from_ckpt: Dict[str, int] = {}
        for k, t in state.items():
            if not (isinstance(k, str) and k.startswith("emb_layers.") and k.endswith(".weight")):
                continue
            name = k[len("emb_layers.") : -len(".weight")]
            if hasattr(t, "shape") and len(t.shape) == 2:
                rows_from_ckpt[name] = int(t.shape[0])
                emb_dims[name] = int(t.shape[1])

        # Sanity: ensure every feature in feature_order has a derived dimension
        missing = [f for f in self.feature_order if f not in emb_dims]
        if missing:
            # Try hparams as a fallback (not always present)
            hp = {}
            for k in ("hyper_parameters", "hparams", "hparams_initial"):
                if k in obj and isinstance(obj[k], dict):
                    hp = obj[k]
                    break
            hp_dims = (hp.get("emb_dims") or {}) if isinstance(hp.get("emb_dims"), dict) else {}
            for f in missing:
                if f in hp_dims:
                    emb_dims[f] = int(hp_dims[f])
                else:
                    raise RuntimeError(f"Could not recover embedding dim for feature '{f}' from checkpoint.")

        # Optional: warn if sidecar cardinalities disagree with ckpt rows
        # This does not have to be fatal (we clamp inputs later), but it's good to know.
        for f in self.feature_order:
            r_ckpt = rows_from_ckpt.get(f)
            if r_ckpt is not None:
                # Your training used exact 'cards[name]' as num_embeddings for that feature.
                r_sidecar = int(self.cards.get(f, r_ckpt))
                if r_sidecar != r_ckpt:
                    # Keep going; inference clamps inputs to [0, r_ckpt-1]
                    print(f"[pop-infer] Warning: rows mismatch for '{f}': sidecar={r_sidecar} vs ckpt={r_ckpt}. Using ckpt rows.")

                # Always trust checkpoint rows at reconstruction time
                self.cards[f] = r_ckpt

        # Rebuild the Lightning model exactly and load weights
        self.model = PopulationNetLit(
            cards=self.cards,
            emb_dims=emb_dims,
            hidden_dims=obj.get("hyper_parameters", {}).get("hidden_dims", [64, 64])
            if isinstance(obj.get("hyper_parameters", {}), dict) else [64, 64],
            lr=float(obj.get("hyper_parameters", {}).get("lr", 1e-3)) if isinstance(obj.get("hyper_parameters", {}), dict) else 1e-3,
            weight_decay=float(obj.get("hyper_parameters", {}).get("weight_decay", 1e-4))
            if isinstance(obj.get("hyper_parameters", {}), dict) else 1e-4,
            dropout=float(obj.get("hyper_parameters", {}).get("dropout", 0.1))
            if isinstance(obj.get("hyper_parameters", {}), dict) else 0.1,
            feature_order=self.feature_order,
            use_soft_labels=True,  # matches how you trained
        ).to(self.device)

        # Some PL exports prefix params; try strict=True then fallback
        try:
            self.model.load_state_dict(state, strict=True)
        except Exception:
            # Remove a possible "model." prefix (rare if LightningModule saved itself directly)
            fixed = {}
            for k, v in state.items():
                fixed[k[6:]] = v if k.startswith("model.") else v
            self.model.load_state_dict(fixed, strict=False)

        self.model.eval()

        # For clamping inputs to valid id ranges
        self._max_idx = {name: (int(self.cards[name]) - 1) for name in self.feature_order}

    def _encode_one(self, feats: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        feats: dict with integer IDs for each categorical feature in feature_order.
               Missing → 0; Out-of-range → clamped to [0, card-1].
        """
        xb: Dict[str, torch.Tensor] = {}
        for name in self.feature_order:
            v = int(feats.get(name, 0))
            vmax = self._max_idx[name]
            if v < 0:
                v = 0
            elif v > vmax:
                v = vmax
            xb[name] = torch.tensor([v], dtype=torch.long, device=self.device)  # [1]
        return xb

    @torch.no_grad()
    def predict_proba(self, feats: Dict[str, int]) -> Dict[str, float]:
        """Return probabilities over ACTIONS for a single row of categorical ids."""
        xb = self._encode_one(feats)
        logits = self.model(xb)  # [1,3]
        p = F.softmax(logits, dim=-1).squeeze(0).tolist()
        return {a: float(p[i]) for i, a in enumerate(ACTIONS)}

    @torch.no_grad()
    def predict_class(self, feats: Dict[str, int]) -> Tuple[str, int]:
        """Return (label, index) for argmax class."""
        xb = self._encode_one(feats)
        idx = int(torch.argmax(self.model(xb), dim=-1).item())
        return ACTIONS[idx], idx
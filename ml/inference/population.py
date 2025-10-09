from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from ml.models.population_net import PopulationNetLit
from ml.utils.sidecar import load_sidecar


ACTIONS = ["FOLD", "CALL", "RAISE"]  # p_fold/p_call/p_raise order

class PopulationNetInference:
    """
    Inference for PopulationNetLit checkpoints.

    - Uses the sidecar for feature_order + cards (and clamps inputs into valid id ranges).
    - Recovers exact embedding dims from the checkpoint tensors.
    - Accepts integer IDs for all categorical features (same IDs as your training parquet).
    """

    def __init__(self, ckpt_path: str | Path, device: Optional[torch.device] = None):
        import torch
        import torch.nn.functional as F  # noqa: F401  (for completeness if used externally)

        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        # ---- sidecar (prefer “<ckpt>.sidecar.json”) ----
        sidecar_path = self.ckpt_path.with_suffix(self.ckpt_path.suffix + ".sidecar.json")
        if not sidecar_path.exists():
            # tolerant fallback names in the same directory
            for name in ("best_sidecar.json", "sidecar.json"):
                cand = self.ckpt_path.parent / name
                if cand.exists():
                    sidecar_path = cand
                    break
        sc = load_sidecar(sidecar_path)

        self.feature_order: List[str] = list(sc["feature_order"])
        self.cards: Dict[str, int] = {str(k): int(v) for k, v in sc["cards"].items()}

        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))

        # ---- load checkpoint dict ----
        obj = torch.load(self.ckpt_path, map_location=self.device)
        if not isinstance(obj, dict):
            raise RuntimeError("Unexpected checkpoint format; expected a dict-like object.")

        # Prefer Lightning’s "state_dict" (with module keys)
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state: Dict[str, torch.Tensor] = obj["state_dict"]
            hp = obj.get("hyper_parameters", {}) if isinstance(obj.get("hyper_parameters"), dict) else {}
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state = obj["model_state_dict"]
            hp = obj.get("hyper_parameters", {}) if isinstance(obj.get("hyper_parameters"), dict) else {}
        else:
            # Sometimes the whole obj is the state_dict
            state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            hp = obj.get("hyper_parameters", {}) if isinstance(obj.get("hyper_parameters"), dict) else {}

        # ---- recover embedding dims from tensors ----
        emb_dims: Dict[str, int] = {}
        rows_from_ckpt: Dict[str, int] = {}
        for k, t in state.items():
            if isinstance(k, str) and k.startswith("emb_layers.") and k.endswith(".weight"):
                name = k[len("emb_layers."):-len(".weight")]
                if hasattr(t, "shape") and len(t.shape) == 2:
                    rows_from_ckpt[name] = int(t.shape[0])
                    emb_dims[name] = int(t.shape[1])

        # Fill any missing dims from hparams if present
        missing = [f for f in self.feature_order if f not in emb_dims]
        hp_dims = hp.get("emb_dims") or {}
        if isinstance(hp_dims, dict):
            for f in list(missing):
                if f in hp_dims:
                    emb_dims[f] = int(hp_dims[f])
                    missing.remove(f)
        if missing:
            raise RuntimeError(f"Could not recover embedding dim(s) for: {missing}")

        # Trust checkpoint rows over sidecar if they disagree (we’ll clamp inputs anyway)
        for f in self.feature_order:
            r_ckpt = rows_from_ckpt.get(f)
            if r_ckpt is not None:
                self.cards[f] = r_ckpt

        # ---- rebuild model and load weights ----
        self.model = PopulationNetLit(
            cards=self.cards,
            emb_dims=emb_dims,
            hidden_dims=hp.get("hidden_dims", [64, 64]) if isinstance(hp, dict) else [64, 64],
            lr=float(hp.get("lr", 1e-3)) if isinstance(hp, dict) else 1e-3,
            weight_decay=float(hp.get("weight_decay", 1e-4)) if isinstance(hp, dict) else 1e-4,
            dropout=float(hp.get("dropout", 0.1)) if isinstance(hp, dict) else 0.1,
            feature_order=self.feature_order,
            use_soft_labels=True,
        ).to(self.device)

        try:
            self.model.load_state_dict(state, strict=True)
        except Exception:
            # lenient: strip an optional "model." prefix if present
            fixed = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }
            self.model.load_state_dict(fixed, strict=False)

        self.model.eval()
        self._max_idx = {name: (int(self.cards[name]) - 1) for name in self.feature_order}

    # ---------- convenience constructors ----------

    @classmethod
    def from_dir(cls, dir_path: str | Path, ckpt_name: str = "best.ckpt") -> "PopulationNetInference":
        """
        Load from a directory that contains a checkpoint + sidecar.
        Tries (in order):
          - <dir>/<ckpt_name>
          - the newest *.ckpt in <dir>
        """
        d = Path(dir_path)
        ckpt = d / ckpt_name
        if not ckpt.exists():
            # pick newest *.ckpt
            cands = sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                raise FileNotFoundError(f"No checkpoints found in {d}")
            ckpt = cands[0]
        return cls(ckpt_path=ckpt)

    # ---------- encoding & prediction ----------

    def _encode_one(self, feats: Dict[str, int]) -> Dict[str, torch.Tensor]:
        xb: Dict[str, torch.Tensor] = {}
        for name in self.feature_order:
            v = int(feats.get(name, 0))
            vmax = self._max_idx[name]
            if v < 0: v = 0
            if v > vmax: v = vmax
            xb[name] = torch.tensor([v], dtype=torch.long, device=self.device)  # [1]
        return xb

    @torch.no_grad()
    def predict_proba(self, feats: Dict[str, int]) -> Dict[str, float]:
        xb = self._encode_one(feats)
        logits = self.model(xb)  # [1,3]
        p = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return {ACTIONS[i]: float(p[i]) for i in range(3)}

    @torch.no_grad()
    def predict_class(self, feats: Dict[str, int]) -> Tuple[str, int]:
        xb = self._encode_one(feats)
        idx = int(torch.argmax(self.model(xb), dim=-1).item())
        return ACTIONS[idx], idx

    @torch.no_grad()
    def predict_proba_many(self, rows: List[Dict[str, int]]) -> List[List[float]]:
        """
        Batched probabilities for a list of feature dicts.
        Returns [[p_fold,p_call,p_raise], ...]
        """
        if not rows:
            return []
        # stack as batch dict
        batch: Dict[str, List[int]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                v = int(r.get(k, 0))
                vmax = self._max_idx[k]
                if v < 0: v = 0
                if v > vmax: v = vmax
                batch[k].append(v)
        xb = {k: torch.tensor(vs, dtype=torch.long, device=self.device) for k, vs in batch.items()}
        logits = self.model(xb)  # [B,3]
        return torch.softmax(logits, dim=-1).tolist()
from __future__ import annotations
from pathlib import Path
from typing import Union, Sequence, Dict, Optional, Any, List
import torch.nn.functional as F
import numpy as np
import torch
from ml.features.hands import hand_to_169_label
from ml.inference.deduce_context import deduce_preflop_context
from ml.inference.policy.types import PolicyRequest, PolicyResponse, Action
from ml.models.preflop_rangenet import RangeNetLit
from ml.utils.sidecar import load_sidecar

DeviceLike = Union[str, torch.device]

def _to_device(device: DeviceLike = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))

class PreflopPolicy:
    """
    Single, self-contained preflop policy that:
      - loads the trained RangeNet (Lightning) from ckpt(+sidecar)
      - encodes a single inference row using sidecar id_maps/cards
      - predicts a simple action distribution (with optional equity nudge)
    """

    def __init__(
        self,
        *,
        model: RangeNetLit,
        feature_order: Sequence[str],
        cards: Dict[str, int],
        id_maps: Dict[str, Dict[str, int]] | None,
        device: torch.device,
    ):
        self.model = model.eval().to(device)
        self.device = device

        self.feature_order = list(feature_order)
        self.cards = {k: int(v) for k, v in cards.items()}
        self.id_maps = {k: {str(a): int(b) for a, b in m.items()} for k, m in (id_maps or {}).items()}

    # ---------- loaders ----------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar_path: Union[str, Path],
        *,
        device: DeviceLike = "auto",
    ) -> "PreflopPolicy":
        dev = _to_device(device)
        sc = load_sidecar(sidecar_path)
        feature_order = sc["feature_order"]
        cards = {str(k): int(v) for k, v in sc["cards"].items()}
        id_maps = sc.get("id_maps") or {}

        # ⚠️ RangeNetLit needs cards/feature_order passed to constructor
        model = RangeNetLit.load_from_checkpoint(
            str(checkpoint_path),
            map_location=dev,
            cards=cards,
            feature_order=feature_order,
        )
        model.eval().to(dev)

        return cls(
            model=model,
            feature_order=feature_order,
            cards=cards,
            id_maps=id_maps,
            device=dev,
        )

    @classmethod
    def from_dir(
        cls,
        ckpt_dir: Union[str, Path],
        ckpt_name: Optional[str] = None,
        device: DeviceLike = "auto",
    ) -> "PreflopPolicy":
        d = Path(ckpt_dir)
        if ckpt_name is None:
            cands = sorted(d.glob("range_preflop-*-*.ckpt"))
            ckpt = cands[0] if cands else (d / "best.ckpt")
        else:
            ckpt = d / ckpt_name
        sidecar = d / "best_sidecar.json"
        return cls.from_checkpoint(ckpt, sidecar, device=device)

    # ---------- encode one row ----------
    def _unknown_idx(self, feat: str) -> int:
        C = int(self.cards.get(feat, 1))
        return max(C - 1, 0)

    def _encode_row(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for feat in self.feature_order:
            enc = self.id_maps.get(feat, {})
            card = int(self.cards.get(feat, max(len(enc), 1)))
            unk = min(max(len(enc), 1) - 1, card - 1) if enc else self._unknown_idx(feat)

            val = row.get(feat, None)
            key = "__NONE__" if val is None else str(val).upper()
            idx = enc.get(key, unk)
            if idx >= card:
                idx = card - 1
            out[feat] = torch.tensor([int(idx)], dtype=torch.long, device=self.device)
        return out

    # ---------- helpers ----------
    @staticmethod
    def _softmax_np(logits: np.ndarray, T: float) -> List[float]:
        z = logits.astype("float32") / max(T, 1e-6)
        z -= np.max(z)
        p = np.exp(z)
        s = float(p.sum())
        return (p / s).tolist() if s > 0 else (np.ones_like(p) / len(p)).tolist()

    @torch.no_grad()
    def predict(
            self,
            req: PolicyRequest,
            *,
            equity: Optional[Dict[str, float]] = None,  # {"p_win","p_tie","p_lose"}
            temperature: float = 1.0,
            equity_nudge: float = 0.0,  # tiny tilt only
    ) -> PolicyResponse:
        # -------- 1) derive minimal context (hero-centric) --------
        stack = float(req.eff_stack_bb or req.pot_bb or 100.0)

        ctx_res = deduce_preflop_context(
            hero_pos=req.hero_pos,
            villain_pos=req.villain_pos,
            actions_hist=req.actions_hist,
            raw=req.raw,
            pot_bb=req.pot_bb,
        )
        hero_pos = ctx_res["hero_pos"]
        opener_pos = ctx_res["opener_pos"]
        opener_action = ctx_res["opener_action"]
        facing_open = bool(ctx_res["facing_open"])

        # Build model row (case-insensitive tolerance)
        row_raw = {
            "STACK_BB": stack,  # if categorical in training, sidecar id_maps must contain bins
            "HERO_POS": hero_pos,
            "OPENER_POS": opener_pos,
            "OPENER_ACTION": opener_action,
            "CTX": (req.raw.get("ctx") or "SRP").upper(),
        }
        row = {k.upper(): v for k, v in {k.lower(): v for k, v in row_raw.items()}.items()}

        # -------- 2) model → villain 169 range --------
        xb = self._encode_row(row)  # expects dict aligned to feature_order
        logits = self.model(xb)  # [1,169]
        rng = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()  # [169]
        s = float(rng.sum())
        rng = (rng / s).astype("float32") if s > 1e-8 else (np.ones(169, dtype="float32") / 169.0)

        hero_mass = None
        if req.hero_hand:
            try:
                hid = hand_to_169_label(req.hero_hand)  # 0..168
                if 0 <= int(hid) < 169:
                    hero_mass = float(rng[int(hid)])
            except Exception:
                pass

        # -------- 3) action prior (+ tiny equity tilt) --------
        if facing_open:
            tokens = ["FOLD", "CALL", "RAISE_300"]  # F / flat / 3-bet ~3x
            base = np.array([0.35, 0.40, 0.25], dtype="float32")  # prior mass
        else:
            tokens = ["FOLD", "RAISE_250"]  # open-fold vs open to 2.5bb
            base = np.array([0.30, 0.70], dtype="float32")

        # keep a copy before applying equity (for diagnostics)
        base_before = base.copy()

        # apply tiny equity nudge (stable + normalized)
        if equity and equity_nudge > 0:
            p_win = float(equity.get("p_win", 0.5))
            tilt = (p_win - 0.5) * float(equity_nudge)
            if facing_open and "CALL" in tokens:
                i_call = tokens.index("CALL")
                base[i_call] = max(0.0, base[i_call] + tilt)
            elif (not facing_open) and "RAISE_250" in tokens:
                i_r = tokens.index("RAISE_250")
                base[i_r] = max(0.0, base[i_r] + tilt)
            s = float(base.sum())
            if s > 0:
                base = base / s

        # diagnostic: magnitude of equity effect in (log) prior space
        log_before = np.log(base_before + 1e-12)
        log_after = np.log(base + 1e-12)
        delta_eq_l1 = float(np.abs(log_after - log_before).sum())

        # temperature → probs
        probs = self._softmax_np(np.log(base + 1e-8), temperature)

        return PolicyResponse(
            actions=tokens,
            probs=probs,
            evs=[0.0] * len(tokens),
            notes=[f"preflop policy (T={temperature:.2f}, eq_nudge={equity_nudge:.3f})"],
            debug={
                "street": 0,
                "facing_open": facing_open,
                "input_row": row_raw,
                "villain_range_169_sum": float(rng.sum()),
                "hero_prior_mass_in_villain_range": hero_mass,
                "equity": equity or {},
                "delta_eq_l1": delta_eq_l1,  # <- equity influence diagnostic
            },
        )
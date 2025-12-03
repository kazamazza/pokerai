from __future__ import annotations
from pathlib import Path
from typing import Union, Sequence, Dict, Optional, Any, List
import torch.nn.functional as F
import numpy as np
import torch
from ml.features.hands import hand_to_169_label
from ml.inference.deduce_context import deduce_preflop_context
from ml.inference.ev_calculator import EVCalculator
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

    def _dynamic_base_prior(self, tokens: list[str], req: PolicyRequest,
                            evs: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generates base prior for each action based on stack size, facing, and optionally EVs.
        """
        stack = float(req.eff_stack_bb or req.pot_bb or 100.0)
        facing = bool(req.facing_bet)
        base = []

        for a in tokens:
            if a == "FOLD":
                base.append(0.20 if facing else 0.01)
            elif a == "CALL":
                base.append(0.40 if facing else 0.0)
            elif a == "CHECK":
                base.append(0.40 if not facing else 0.0)
            elif a.startswith("RAISE_") or a.startswith("OPEN_"):
                try:
                    amt = float(a.split("_")[1])
                    frac = amt / stack
                    score = 1.0 - abs(frac - 0.5)  # favor mid-stacks
                    if evs and a in evs:
                        score *= max(evs[a], 0.0) + 1.0  # weight by EV if available
                    base.append(score)
                except:
                    base.append(0.01)
            else:
                base.append(0.01)

        base = np.array(base, dtype="float32")
        if base.sum() > 0:
            base /= base.sum()
        return base

    def _legal_mask(self, tokens: list[str], req: PolicyRequest) -> np.ndarray:
        stack = float(req.eff_stack_bb or req.pot_bb or 100.0)
        facing = bool(req.facing_bet)
        faced_frac = float(req.faced_size_frac or 0.0)
        legal = []

        for a in tokens:
            if a == "FOLD":
                legal.append(1.0 if facing else 0.0)
            elif a == "CHECK":
                legal.append(1.0 if not facing else 0.0)
            elif a == "CALL":
                legal.append(1.0 if facing and faced_frac > 0 else 0.0)
            elif a.startswith("RAISE_") or a.startswith("OPEN_"):
                try:
                    val = float(a.split("_")[1])
                    legal.append(1.0 if val < stack * 2 else 0.0)  # allow if not all-in
                except:
                    legal.append(0.0)
            else:
                legal.append(0.0)
        return np.array(legal, dtype="float32")

    @torch.no_grad()
    def predict(
            self,
            req: PolicyRequest,
            *,
            equity: Optional[Dict[str, float]] = None,
            temperature: float = 1.0,
            equity_nudge: float = 0.0,
    ) -> PolicyResponse:
        # -------- 1) derive preflop context --------
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

        # -------- 2) encode input row for RangeNet --------
        row_raw = {
            "STACK_BB": stack,
            "HERO_POS": hero_pos,
            "OPENER_POS": opener_pos,
            "OPENER_ACTION": opener_action,
            "CTX": (req.raw.get("ctx") or "SRP").upper(),
        }
        row = {k.upper(): v for k, v in {k.lower(): v for k, v in row_raw.items()}.items()}
        xb = self._encode_row(row)

        logits = self.model(xb)  # [1, 169]
        rng = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        rng = (rng / float(rng.sum())) if rng.sum() > 1e-8 else np.ones(169, dtype="float32") / 169.0

        hero_mass = None
        if req.hero_hand:
            try:
                hid = hand_to_169_label(req.hero_hand)
                hero_mass = float(rng[int(hid)])
            except Exception:
                pass

        # -------- 3) define action space --------
        if facing_open:
            tokens = ["FOLD", "CALL", "RAISE_200", "RAISE_300"]
        else:
            tokens = ["FOLD", "OPEN_200", "OPEN_300"]

        # -------- 4) compute EVs --------
        ev_sig = EVCalculator().compute(req)
        evs = ev_sig.evs if ev_sig and ev_sig.available else {}

        # -------- 5) dynamic base prior --------
        base_before = self._dynamic_base_prior(tokens, req, evs=evs)
        base = base_before.copy()

        # -------- 6) optional equity nudge --------
        if equity and equity_nudge > 0:
            p_win = float(equity.get("p_win", 0.5))
            tilt = (p_win - 0.5) * equity_nudge
            if facing_open and "CALL" in tokens:
                base[tokens.index("CALL")] += tilt
            elif not facing_open and "OPEN_200" in tokens:
                base[tokens.index("OPEN_200")] += tilt
            base = np.clip(base, 0, 1)
            if base.sum() > 0:
                base /= base.sum()

        log_base = np.log(base + 1e-12)
        log_base_before = np.log(base_before + 1e-12)
        delta_eq_l1 = float(np.abs(log_base - log_base_before).sum())

        # -------- 7) apply temperature --------
        probs = self._softmax_np(log_base, temperature)

        # -------- 8) legality mask --------
        legal_mask = self._legal_mask(tokens, req)
        if legal_mask.sum() > 0:
            probs = probs * legal_mask
            probs = probs / probs.sum()  # re-normalize

        # -------- 9) finalize --------
        best_idx = int(np.argmax(probs))
        best_action = tokens[best_idx]

        return PolicyResponse(
            actions=tokens,
            probs=probs,
            evs=[evs.get(a, 0.0) for a in tokens],
            best_action=best_action,
            notes=[f"preflop policy (T={temperature:.2f}, eq_nudge={equity_nudge:.3f})"],
            debug={
                "street": 0,
                "facing_open": facing_open,
                "input_row": row_raw,
                "villain_range_169_sum": float(rng.sum()),
                "hero_prior_mass_in_villain_range": hero_mass,
                "equity": equity or {},
                "delta_eq_l1": delta_eq_l1,
                "evs": evs,
            },
        )
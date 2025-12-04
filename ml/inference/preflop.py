from __future__ import annotations
from pathlib import Path
from typing import Union, Sequence, Dict, Optional, Any, List
import torch.nn.functional as F
import numpy as np
import torch
from ml.inference.deduce_context import deduce_preflop_context
from ml.inference.policy.types import PolicyRequest
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
    def predict(self, req: PolicyRequest) -> np.ndarray:
        """
        PURE RANGENET inference.
        Returns a 169-dim probability vector.
        Includes full debug prints to verify:
            - deduced context
            - raw row values
            - encoded categorical IDs
            - logits
            - softmax
        """

        print("\n================= PREFLOP RANGE DEBUG =================")

        # ----- Step 1: Basic fields -----
        stack = float(req.eff_stack_bb or req.pot_bb or 100.0)
        print("[stack_bb]", stack)

        # ----- Step 2: Deduce context -----
        ctx_res = deduce_preflop_context(
            hero_pos=req.hero_pos,
            villain_pos=req.villain_pos,
            actions_hist=req.actions_hist,
            raw=req.raw,
            pot_bb=req.pot_bb,
        )
        print("[deduce_preflop_context]", ctx_res)

        # ----- Step 3: Build raw row exactly matching x_cols -----
        row_raw = {
            "STACK_BB": stack,
            "HERO_POS": ctx_res["hero_pos"],
            "OPENER_POS": ctx_res["opener_pos"],
            "OPENER_ACTION": ctx_res["opener_action"],
            "CTX": (req.raw.get("ctx") or "SRP").upper(),
        }
        print("[row_raw]", row_raw)

        # Normalize to UPPER-case keys (expected by id_maps/sidecar)
        row = {k.upper(): v for k, v in {k.lower(): v for k, v in row_raw.items()}.items()}
        print("[row_normalized]", row)

        # ----- Step 4: Encode using sidecar id_maps -----
        xb = self._encode_row(row)

        print("[encoded_row IDs]")
        for feat in self.feature_order:
            tid = xb[feat].item()
            print(f"   {feat}: {row.get(feat)}  ->  id={tid}")

        # ----- Step 5: Forward through model -----
        logits = self.model(xb)  # shape [1,169]
        logits_np = logits.detach().cpu().numpy().squeeze()

        print("[logits first 10]", np.round(logits_np[:10], 4))

        # ----- Step 6: Softmax -----
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        probs = probs / probs.sum() if probs.sum() > 1e-8 else np.ones(169) / 169.0

        print("[probs first 10]", np.round(probs[:10], 5))
        print("[probs sum]", probs.sum())
        print("=======================================================\n")

        return probs
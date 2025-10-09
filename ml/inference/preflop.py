from __future__ import annotations
from pathlib import Path
import torch
from ml.features.hands import hand_to_169_label
from ml.inference.equity import EquityNetInfer
from ml.inference.policy.types import PolicyRequest, PolicyResponse, Action
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
import numpy as np
from ml.models.preflop_rangenet import RangeNetLit
from ml.utils.device import DeviceLike
from ml.utils.sidecar import load_sidecar

def _to_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

@dataclass
class PreflopDeps:
    range_pre: Any
    equity: Optional[Any] = None

class PreflopPolicy:
    """
    Orchestrates preflop decision using:
      - Range model (villain 169-range prior)
      - Optional equity model (small guidance)
    Returns a normalized PolicyResponse.
    """
    def __init__(self, deps: PreflopDeps):
        if deps.range_pre is None:
            raise ValueError("PreflopPolicy requires range_pre")
        self.rng = deps.range_pre
        self.eq  = deps.equity

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: str | Path,
            sidecar_path: str | Path,
            *,
            equity_dir: str | Path | None = None,
            device: str = "auto",
    ) -> "PreflopPolicy":
        dev = _to_device(device)
        sc = load_sidecar(sidecar_path)
        cards = {str(k): int(v) for k, v in sc["cards"].items()}
        feature_order = list(sc["feature_order"])

        model = RangeNetLit.load_from_checkpoint(
            str(checkpoint_path),
            map_location=dev,
            cards=cards,
            feature_order=feature_order,
        )
        model.eval().to(dev)

        eq = None
        if equity_dir:
            try:
                eq = EquityNetInfer.from_dir(equity_dir, device=dev.type)
            except Exception as e:
                print(f"[warn] failed to load EquityNet from {equity_dir}: {e}")

        return cls(PreflopDeps(range_pre=model, equity=eq))

    @classmethod
    def from_dir(
            cls,
            range_dir: Union[str, Path],
            *,
            equity_dir: Union[str, Path, None] = None,
            device: DeviceLike = "auto",
    ) -> "PreflopPolicy":
        """
        Discover best/last checkpoint + sidecar in range_dir, then delegate to from_checkpoint.
        Mirrors the Postflop loader behavior.
        """
        rdir = Path(range_dir)
        if not rdir.exists():
            raise FileNotFoundError(f"Preflop range model dir not found: {rdir}")

        # Prefer pattern, then best/last, else any .ckpt
        cands = sorted(rdir.glob("range_preflop-*-*.ckpt"))
        if cands:
            ckpt_path = cands[0]
        elif (rdir / "best.ckpt").exists():
            ckpt_path = rdir / "best.ckpt"
        elif (rdir / "last.ckpt").exists():
            ckpt_path = rdir / "last.ckpt"
        else:
            any_ckpt = sorted(rdir.glob("*.ckpt"))
            if not any_ckpt:
                raise FileNotFoundError(f"No checkpoint found in {rdir}")
            ckpt_path = any_ckpt[0]

        # Sidecar preference: best_sidecar.json → sidecar.json → <ckpt>.sidecar.json
        sidecar_path = (
            rdir / "best_sidecar.json"
            if (rdir / "best_sidecar.json").exists()
            else (rdir / "sidecar.json"
                  if (rdir / "sidecar.json").exists()
                  else ckpt_path.with_suffix(ckpt_path.suffix + ".sidecar.json"))
        )
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Sidecar JSON not found: {sidecar_path}")

        return cls.from_checkpoint(
            ckpt_path,
            sidecar_path,
            equity_dir=equity_dir,
            device=device,
        )


    def _encode_action(self, a: Action) -> str:
        k = str(a.kind).upper()
        if k in ("FOLD", "CHECK", "CALL", "ALLIN"):
            return k
        if k == "BET":
            if a.size_pct is not None:
                pct = int(round(a.size_pct))
                bucket = min([25, 33, 50, 66, 75, 100], key=lambda x: abs(x - pct))
                return f"BET_{bucket}"
            return "BET_50" if a.size_bb else "BET_33"
        if k == "RAISE":
            if a.size_mult is not None:
                m = float(a.size_mult)
                if abs(m - 1.5) < 0.11: return "RAISE_150"
                if abs(m - 2.0) < 0.11: return "RAISE_200"
                if abs(m - 3.0) < 0.11: return "RAISE_300"
                if abs(m - 4.0) < 0.11: return "RAISE_400"
                if abs(m - 5.0) < 0.11: return "RAISE_500"
                return "RAISE_200"
            if a.size_pct is not None:
                return f"RAISE_{int(round(a.size_pct))}"
            return "RAISE_200"
        return k

    # --- small helpers ---
    @staticmethod
    def _safe_np(x) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype="float32")

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        z = (logits / temperature).astype("float32")
        z = z - np.max(z)  # stability
        p = np.exp(z)
        s = p.sum()
        return p / s if s > 0 else np.ones_like(p) / len(p)

    @staticmethod
    def _norm_probs(p: Sequence[float]) -> List[float]:
        p = np.clip(np.asarray(p, dtype="float32"), 0.0, None)
        s = float(p.sum())
        return (p / s).tolist() if s > 0 else (np.ones_like(p) / len(p)).tolist()

    def _legal_actions(self, facing_open: bool) -> List[Action]:
        if facing_open:
            return [Action("FOLD"), Action("CALL"), Action("RAISE", size_mult=3.0)]
        # opening
        return [Action("FOLD"), Action("RAISE", size_bb=2.5)]

    def predict(
        self,
        req: PolicyRequest,
        *,
        temperature: float = 1.0,
        equity_nudge: float = 0.0,  # 0..0.1 small prior tilt; keep tiny
    ) -> PolicyResponse:
        stack = float(req.stack_bb or req.eff_stack_bb or 100.0)
        row = {
            "stack_bb": stack,
            "hero_pos": (req.hero_pos or "").upper(),
            "opener_pos": (req.opener_pos or "").upper(),
            "opener_action": (req.opener_action or "RAISE").upper(),
            "ctx": (req.ctx or "SRP").upper(),
        }
        rng = self._safe_np(self.rng.predict_proba([row]))
        if rng.ndim != 2 or rng.shape[0] != 1 or rng.shape[1] != 169:
            raise ValueError(f"range_pre.predict_proba must return [1,169], got {tuple(rng.shape)}")
        rng_169 = rng[0].astype("float32")
        rng_169 = rng_169 / max(rng_169.sum(), 1e-8)
        hero_mass = None
        if req.hero_hand:
            try:
                idx = getattr(self.rng, "hand_to_id", None)
                if isinstance(idx, dict):
                    hid = idx.get(hand_to_169_label(req.hero_hand))
                    if hid is not None and 0 <= hid < 169:
                        hero_mass = float(rng_169[int(hid)])
            except Exception:
                pass

        equity = {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}
        if self.eq and req.hero_hand:
            try:
                out = self.eq.predict([{"street": 0, "hand_id": hand_to_169_label(req.hero_hand)}])
                if out:
                    p_win, p_tie, p_lose = out[0]
                    equity = {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}
            except Exception:
                pass
        facing_open = bool(req.facing_open) or (
            row["opener_action"] == "RAISE" and row["hero_pos"] in ("BB", "SB")
        )
        actions = self._legal_actions(facing_open)
        tokens  = [self._encode_action(a) for a in actions]
        if facing_open:
            base = np.array([0.35, 0.40, 0.25], dtype="float32")  # FOLD/CALL/RAISE3x
        else:
            base = np.array([0.30, 0.70], dtype="float32")        # FOLD/RAISE2.5bb

        if equity_nudge > 0 and "CALL" in tokens:
            i_call  = tokens.index("CALL")
            tilt = (equity["p_win"] - 0.5) * equity_nudge
            base[i_call] = max(0.0, base[i_call] + tilt)
            base = base / base.sum()

        probs = self._softmax(np.log(base + 1e-8), temperature)
        probs = self._norm_probs(probs)

        return PolicyResponse(
            actions=tokens,
            probs=probs,
            evs=[0.0] * len(tokens),
            notes=[f"preflop stub; temp={temperature}, equity_nudge={equity_nudge}"],
            debug={
                "street": 0,
                "input_row": row,
                "villain_range_169": rng_169.tolist(),
                "hero_prior_mass_in_villain_range": hero_mass,
                "equity": equity,
                "facing_open": facing_open,
            },
        )
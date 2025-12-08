from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from ml.features.hands import hand_to_169_label, hand169_id_from_hand_code
from ml.inference.policy.types import PolicyRequest


@dataclass
class EquitySig:
    available: bool
    p_win: float = 0.5
    p_tie: float = 0.0
    p_lose: float = 0.5
    err: Optional[str] = None

@dataclass
class ExploitSig:
    available: bool
    raw: Optional[np.ndarray] = None          # shape (3,) F/C/R logit deltas
    probs: Optional[Tuple[float,float,float]] = None  # softmax(raw)
    counts_total: float = 0.0
    prior: Optional[Tuple[float,float,float]] = None
    err: Optional[str] = None

@dataclass
class FacingInfo:
    is_facing: bool
    size_frac: Optional[float] = None

@dataclass
class RootInfo:
    is_root: bool
    bet_menu: Optional[List[float]] = None


# --- 169 helpers (no external HAND169_TO_ID required) -----------------------
_FALLBACK_169_MAP: dict[str, int] | None = None
_RANKS = "AKQJT98765432"

def _build_canonical_169_map() -> dict[str, int]:
    labels: list[str] = []
    # Pairs: AA..22
    for r in _RANKS:
        labels.append(r + r)
    # hi>lo: suited then offsuit
    for i, hi in enumerate(_RANKS):
        for lo in _RANKS[i+1:]:
            labels.append(hi + lo + "s")
            labels.append(hi + lo + "o")
    assert len(labels) == 169
    return {lab: i for i, lab in enumerate(labels)}

def _label_to_169_id(label: str) -> int | None:
    """Return index in HANDS_169 if present, else canonical fallback."""
    # Try user’s grid first (keeps consistency with training)
    try:
        from  ml.features.hands import HANDS_169  # if you keep it somewhere else, adjust import
    except Exception:
        HANDS_169 = None

    if label:
        if HANDS_169:
            try:
                return int(HANDS_169.index(label))
            except Exception:
                pass
        global _FALLBACK_169_MAP
        if _FALLBACK_169_MAP is None:
            _FALLBACK_169_MAP = _build_canonical_169_map()
        return _FALLBACK_169_MAP.get(label)
    return None
# ---------------------------------------------------------------------------


class SignalCollector:
    def __init__(self, eq_model, expl_store, pop_model, router=None, router_facing=None):
        self.eq = eq_model
        self.expl = expl_store
        self.pop = pop_model
        self.router = router
        self.router_facing = router_facing or (getattr(router, "facing", None) if router is not None else None)

    def _coerce_row3(self, out) -> tuple[float, float, float]:
        import numpy as np
        try:
            import torch
        except Exception:
            torch = None

        # Accept: list, tuple, np.ndarray, torch.Tensor; shapes (3,) or (1,3)
        if torch is not None and isinstance(out, torch.Tensor):
            x = out.detach().cpu().numpy()
        elif isinstance(out, np.ndarray):
            x = out
        elif isinstance(out, (list, tuple)):
            x = np.asarray(out)
        else:
            raise ValueError(f"probs3 unsupported type: {type(out)}")

        if x.ndim == 1 and x.shape[0] == 3:
            a, b, c = x
        elif x.ndim == 2 and x.shape == (1, 3):
            a, b, c = x[0]
        else:
            # Some libs return list[[...]]
            if isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], (list, tuple)) and len(out[0]) == 3:
                a, b, c = out[0]
            else:
                raise ValueError(f"expected (3,) or (1,3), got shape {getattr(x,'shape',None)}")
        return float(a), float(b), float(c)

    def collect_equity(self, req) -> EquitySig:
        try:
            if not self.eq:
                return EquitySig(False, err="no_eq_model")
            if not getattr(req, "hero_hand", None):
                return EquitySig(False, err="no_hero_hand")

            # build 169 id (robust, no int() on a string label)
            hid = hand169_id_from_hand_code(req.hero_hand)
            if hid is None:
                return EquitySig(False, err="unknown_169_hand")

            pay = {"street": int(getattr(req, "street", 1)), "hand_id": int(hid)}
            # If your equity model accepts board, pass it (harmless if ignored).
            if getattr(req, "board", None):
                pay["board"] = req.board

            raw = self.eq.predict_proba([pay]) if hasattr(self.eq, "predict_proba") else self.eq.predict([pay])
            p_win, p_tie, p_lose = self._coerce_row3(raw)
            return EquitySig(True, p_win, p_tie, p_lose, None)
        except Exception as e:
            return EquitySig(False, err=f"eq_error:{e}")

    def collect_exploit(self, req: PolicyRequest) -> ExploitSig:
        try:
            # Ensure components exist
            if not self.expl or not self.pop or not getattr(req, "villain_id", None):
                return ExploitSig(False, err="missing_model_or_vid")

            pid = str(req.villain_id)
            # Get signal (observes, computes prior, and returns signal + metadata)
            result = self.expl.get_signal_from_request(pid, req, self.pop)

            if result is None:
                return ExploitSig(False, None, None, 0.0, None, None)

            sig3, prior, total = result

            # Convert logit deltas to softmax probabilities
            import torch
            t = torch.tensor(sig3, dtype=torch.float32).view(1, 3)
            pr = torch.softmax(t, dim=-1)[0]
            probs = (float(pr[0]), float(pr[1]), float(pr[2]))

            return ExploitSig(True, sig3, probs, total, prior, None)

        except Exception as e:
            return ExploitSig(False, err=f"expl_error:{e}")

    def collect_facing(self, req, hero_is_ip: bool) -> FacingInfo:
        try:
            rf = self.router_facing
            if rf is not None and hasattr(rf, "infer_facing_and_size"):
                facing_flag, size_frac = rf.infer_facing_and_size(req, hero_is_ip=hero_is_ip)
                return FacingInfo(bool(facing_flag), size_frac)
        except Exception:
            pass
        # Fallback: trust request fields
        fb = bool(getattr(req, "facing_bet", False))
        return FacingInfo(fb, getattr(req, "faced_size_frac", None))


    def collect_root(self, req, hero_is_ip: bool) -> RootInfo:
        try:
            router = self.router  # ← not router_facing!
            if router and hasattr(router, "infer_root_menu"):
                is_root, bet_menu = router.infer_root_menu(req, hero_is_ip=hero_is_ip)
                return RootInfo(is_root=bool(is_root), bet_menu=bet_menu)
        except Exception:
            pass

        # Fallback: assume root if not facing
        if not getattr(req, "facing_bet", False):
            return RootInfo(is_root=True, bet_menu=getattr(req, "bet_sizes", None))
        return RootInfo(is_root=False)
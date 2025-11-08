from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch, numpy as np

from ml.features.hands import hand_to_169_label, HAND169_TO_ID, hand169_label_to_id


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


# --- 169 mapping helpers ---
_FALLBACK_169_MAP = None
_RANKS = "AKQJT98765432"

def _build_canonical_169_map() -> dict[str, int]:
    labels: list[str] = []
    # Pairs: AA, KK, ..., 22
    for r in _RANKS:
        labels.append(r + r)
    # For each distinct hi>lo: suited then offsuit (common canonical order)
    for i, hi in enumerate(_RANKS):
        for lo in _RANKS[i+1:]:
            labels.append(hi + lo + "s")
            labels.append(hi + lo + "o")
    assert len(labels) == 169
    return {lab: i for i, lab in enumerate(labels)}

def _label_to_169_id(label: str) -> int | None:
    global _FALLBACK_169_MAP
    if not label:
        return None
    # Try an app-provided table first, if it exists
    try:
        # If you have this in your codebase, prefer it (keeps consistency with training)

        hid = HAND169_TO_ID.get(label)
        if hid is not None:
            return int(hid)
    except Exception:
        pass
    # Fallback to canonical construction
    if _FALLBACK_169_MAP is None:
        _FALLBACK_169_MAP = _build_canonical_169_map()
    return _FALLBACK_169_MAP.get(label)

class SignalCollector:
    def __init__(self, eq_model, expl_store, pop_model, router_facing=None):
        self.eq = eq_model
        self.expl = expl_store
        self.pop = pop_model
        self.router_facing = router_facing  # optional: pol_post.facing

    def _coerce_row3(self, out) -> tuple[float, float, float]:
        """
        Accepts torch.Tensor/list/np array shaped (3,) or (1,3) and returns 3 floats.
        """
        try:
            import torch, numpy as np
        except Exception:
            torch = None
            np = None

        # torch
        if torch is not None and isinstance(out, torch.Tensor):
            x = out.detach().cpu().numpy()
        # list-of-lists or list
        elif isinstance(out, (list, tuple)):
            x = out
        # numpy
        elif np is not None and isinstance(out, np.ndarray):
            x = out
        else:
            raise ValueError(f"unsupported type for probs3: {type(out)}")

        # squeeze [1,3] -> [3]
        if hasattr(x, "shape"):
            # numpy
            import numpy as np  # type: ignore
            x = np.asarray(x)
            if x.ndim == 2:
                x = x[0]
            if x.shape[0] != 3:
                raise ValueError(f"expected 3 elements, got shape {x.shape}")
            return float(x[0]), float(x[1]), float(x[2])
        else:
            # python lists
            if len(x) == 0:
                raise ValueError("empty output")
            row = x[0] if isinstance(x[0], (list, tuple)) else x
            if len(row) != 3:
                raise ValueError(f"expected 3 elements, got len {len(row)}")
            return float(row[0]), float(row[1]), float(row[2])

    def collect_equity(self, req) -> EquitySig:
        try:
            if not self.eq or not getattr(req, "hero_hand", None) or not getattr(req, "board", None):
                return EquitySig(False, err="missing_model_or_cards")

            # optional safety: reject overlap hero↔board
            try:
                b = [req.board[i:i + 2].upper() for i in range(0, len(req.board), 2)]
                if any(c in (req.hero_hand or "").upper() for c in b):
                    return EquitySig(False, err="hand_board_overlap")
            except Exception:
                pass

            label = hand_to_169_label(req.hero_hand)  # e.g. "T7s"
            hid = hand169_label_to_id(label)
            if not hid and hid != 0:
                return EquitySig(False, err=f"unknown_169_label:{label}")

            payload = {"street": int(getattr(req, "street", 1)), "hand_id": int(hid)}
            raw = self.eq.predict_proba([payload]) if hasattr(self.eq, "predict_proba") else self.eq.predict([payload])
            # --- never use truthiness on tensors ---
            pw, pt, pl = self._coerce_row3(raw)
            return EquitySig(True, pw, pt, pl, None)
        except Exception as e:
            return EquitySig(False, err=f"eq_error:{e}")

    def collect_exploit(self, req) -> ExploitSig:
        try:
            if not self.expl or not getattr(req, "villain_id", None) or not self.pop:
                return ExploitSig(False, err="missing_model_or_vid")
            sk = self.expl.scenario_key_from_req(req)
            # prior from pop model:
            feats = {
                "stakes_id": int((getattr(req,"raw",{}) or {}).get("stakes_id",0)),
                "street_id": int(getattr(req,"street",0) or 0),
                "ctx_id": int((getattr(req,"raw",{}) or {}).get("ctx_id",0)),
                "hero_pos_id": int((getattr(req,"raw",{}) or {}).get("hero_pos_id",0)),
            }
            pri = self.pop.predict_proba(feats)  # {'FOLD':p,'CALL':p,'RAISE':p}
            prior = (float(pri["FOLD"]), float(pri["CALL"]), float(pri["RAISE"]))
            # counts snapshot
            with self.expl._lock:
                n = self.expl._counts[str(req.villain_id)][sk].copy()
            total = float(n.sum())
            sig3 = self.expl.get_signal_from_request(str(req.villain_id), req, self.pop)
            if sig3 is None:
                return ExploitSig(False, None, None, total, prior, None)
            t = torch.tensor(sig3, dtype=torch.float32).view(1,3)
            pr = torch.softmax(t, dim=-1)[0]
            probs = (float(pr[0]), float(pr[1]), float(pr[2]))
            return ExploitSig(True, sig3, probs, total, prior, None)
        except Exception as e:
            return ExploitSig(False, err=f"expl_error:{e}")

    def collect_facing(self, req, hero_is_ip: bool) -> FacingInfo:
        try:
            if self.router_facing is None:
                fb = bool(getattr(req, "facing_bet", False))
                return FacingInfo(fb, getattr(req,"faced_size_frac", None))
            facing_flag, size_frac = self.router_facing.infer_facing_and_size(req, hero_is_ip=hero_is_ip)
            return FacingInfo(bool(facing_flag), size_frac)
        except Exception:
            fb = bool(getattr(req, "facing_bet", False))
            return FacingInfo(fb, getattr(req,"faced_size_frac", None))
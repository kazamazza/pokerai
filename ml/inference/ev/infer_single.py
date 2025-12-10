from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import json
import torch
import torch.nn as nn

# tiny local encoders to avoid reusing policy helpers
def _to_device(dev: Union[str, torch.device, None]) -> torch.device:
    if isinstance(dev, torch.device):
        return dev
    if dev in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(dev))

def _encode_cats_one(
    feature_order: Sequence[str],
    id_maps: Mapping[str, Mapping[str, int]],
    row: Mapping[str, Any],
    device: torch.device,
) -> torch.LongTensor:
    ids: List[int] = []
    for k in feature_order:
        m = id_maps.get(k, {})
        v = str(row.get(k, ""))
        if v not in m:
            # fall back to first key if unknown (sidecar-built maps are closed)
            v = next(iter(m.keys())) if m else ""
        ids.append(int(m.get(v, 0)))
    return torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)

def _encode_cont_one(
    cont_cols: Sequence[str],
    row: Mapping[str, Any],
    device: torch.device,
) -> torch.FloatTensor:
    vec: List[torch.Tensor] = []
    for name in cont_cols:
        if name == "board_mask_52":
            v = row.get("board_mask_52", [0.0] * 52)
            t = torch.tensor(v, dtype=torch.float32, device=device).view(1, 52)
        else:
            t = torch.tensor([float(row.get(name, 0.0))], dtype=torch.float32, device=device).view(1, 1)
        vec.append(t)
    return torch.cat(vec, dim=-1) if vec else torch.zeros(1, 0, device=device)

def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _hand169_id_from_code(hand: str) -> Optional[int]:
    try:
        from ml.features.hands import hand_to_169_label  # your helper
    except Exception:
        return None
    lab = hand_to_169_label(hand)
    try:
        from ml.features.hands import HANDS_169
        return int(HANDS_169.index(lab))
    except Exception:
        return None

def _make_board_mask(board: str) -> List[float]:
    try:
        from ml.utils.board_mask import make_board_mask_52
        return make_board_mask_52(board)
    except Exception:
        return [0.0] * 52


@dataclass
class EVOutput:
    actions: List[str]
    evs: List[float]
    debug: Dict[str, Any]


class EVInferSingle:
    """
    Sidecar-driven EV inferencer for one split:
      - mode ∈ {"preflop", "root", "facing"} controls row assembly
      - returns per-action EV vector in the sidecar's action_vocab order
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        mode: str,  # "preflop" | "root" | "facing"
        x_cols: Sequence[str],
        cont_cols: Sequence[str],
        action_vocab: Sequence[str],
        id_maps: Mapping[str, Mapping[str, int]],
        device: Optional[torch.device] = None,
        clusterer: Optional[Any] = None,  # BoardClusterer
    ):
        self.model = model.eval()
        self.device = _to_device(device)
        self.model.to(self.device)

        m = str(mode or "").lower()
        assert m in ("preflop", "root", "facing"), f"unknown EV mode: {mode}"
        self.mode = m

        self.x_cols = [str(c) for c in x_cols]
        self.cont_cols = [str(c) for c in cont_cols]
        self.action_vocab = list(action_vocab)
        self.id_maps: Dict[str, Dict[str, int]] = {k: {str(a): int(b) for a, b in m.items()} for k, m in id_maps.items()}
        self.clusterer = clusterer

        # sanity: model head width
        with torch.no_grad():
            x_cat = torch.zeros(1, len(self.x_cols), dtype=torch.long, device=self.device)
            x_cont = torch.zeros(1, sum(52 if c == "board_mask_52" else 1 for c in self.cont_cols), device=self.device)
            out = self.model(x_cat, x_cont)
            if int(out.shape[-1]) != len(self.action_vocab):
                raise ValueError(f"EV head width {out.shape[-1]} != len(action_vocab) {len(self.action_vocab)}")

    @classmethod
    def from_checkpoint(
            cls,
            *,
            checkpoint_path: Union[str, Path],
            sidecar_path: Union[str, Path],
            mode: str,
            device: str = "auto",
            clusterer: Optional[Any] = None,
    ) -> "EVInferSingle":
        dev = _to_device(device)
        sc = _load_json(sidecar_path)

        # schema for encoding (we still need this just like Equity)
        x_cols = sc.get("cat_feature_order") or sc.get("x_cols") or sc.get("feature_order") or []
        cont_cols = sc.get("cont_feature_order") or sc.get("cont_cols") or []
        id_maps = sc.get("id_maps") or {}
        action_vocab = list(sc.get("action_vocab") or [])
        if not x_cols or not cont_cols or not action_vocab:
            missing = []
            if not x_cols: missing.append("cat_feature_order/x_cols")
            if not cont_cols: missing.append("cont_feature_order/cont_cols")
            if not action_vocab: missing.append("action_vocab")
            raise ValueError(f"incomplete EV sidecar: {sidecar_path} (missing: {', '.join(missing)})")

        # Clean path: load with saved hparams (EVLit reconstructs its EVNet)
        from ml.models.evnet import EVLit, EVNet, EVNetConfig
        try:
            lit = EVLit.load_from_checkpoint(str(checkpoint_path), map_location=dev).eval().to(dev)
        except TypeError:
            # Backward-compat: older checkpoints without hparams → rebuild from sidecar
            cat_cards = [len((id_maps or {}).get(c, {})) for c in x_cols]

            def _feat_size(n: str) -> int:
                return 52 if str(n) == "board_mask_52" else 1

            cont_dim = int(sum(_feat_size(n) for n in cont_cols))
            # best-effort defaults for older runs
            hidden_dims = [256, 256]
            dropout = 0.10
            max_emb_dim = 32
            arch = sc.get("arch") or {}
            hidden_dims = arch.get("hidden_dims", hidden_dims)
            dropout = arch.get("dropout", dropout)
            max_emb_dim = arch.get("max_emb_dim", max_emb_dim)

            net = EVNet(EVNetConfig(
                cat_cardinalities=cat_cards,
                cont_dim=cont_dim,
                action_vocab=action_vocab,
                hidden_dims=hidden_dims,
                dropout=float(dropout),
                max_emb_dim=int(max_emb_dim),
            ))
            lit = EVLit.load_from_checkpoint(
                str(checkpoint_path), map_location=dev, net=net, lr=1e-3, weight_decay=1e-4
            ).eval().to(dev)

        return cls(
            model=lit,
            x_cols=list(x_cols),
            cont_cols=list(cont_cols),
            id_maps={str(k): {str(a): int(b) for a, b in (mp or {}).items()} for k, mp in (id_maps or {}).items()},
            action_vocab=action_vocab,
            mode=str(mode),
            device=dev,
            clusterer=clusterer,
        )

    # ---------- row assembly helpers ----------
    def _ctx_guard(self, ctx: str) -> str:
        # If your EV sidecars use VS_OPEN instead of BLIND_VS_STEAL, coerce here if needed.
        return "VS_OPEN" if ctx == "BLIND_VS_STEAL" else ctx

    def _cluster_id(self, board: str) -> int:
        if not self.clusterer:
            return 0
        try:
            return int(self.clusterer.predict(board))
        except Exception:
            return 0

    def _infer_facing(self, req, hero_is_ip: bool) -> Tuple[bool, Optional[float]]:
        try:
            from ml.inference.postflop_single.facing import infer_facing_and_size
            f, s, _ = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
            return bool(f), s
        except Exception:
            fb = bool(getattr(req, "facing_bet", False))
            frac = getattr(req, "faced_size_frac", None)
            if frac is None and getattr(req, "faced_size_pct", None) is not None:
                frac = float(getattr(req, "faced_size_pct")) / 100.0
            return fb, (float(frac) if fb and frac is not None else None)

    # ---------- public ----------
    @torch.no_grad()
    def predict(self, req) -> EVOutput:
        """
        Returns EVOutput with EVs aligned to self.action_vocab.
        Expects req to carry the same info you already pass to policy inference.
        """
        # --- derive minimal fields from req (shared) ---
        street = int(getattr(req, "street", 0) or 0)
        hpos = str(getattr(req, "hero_pos", "") or "").upper()
        vpos = str(getattr(req, "villain_pos", "") or "").upper()

        # flop IP detector (same as policy; BTN IP vs blinds; else later seat is IP)
        if street == 0:
            hero_is_ip = False  # preflop doesn't need IP/OOP here
        else:
            if hpos == "BTN" and vpos in ("SB", "BB"):
                hero_is_ip = True
            elif {hpos, vpos} == {"SB", "BB"}:
                hero_is_ip = (hpos == "BB")
            else:
                order = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
                try: hero_is_ip = order.index(hpos) > order.index(vpos)
                except ValueError: hero_is_ip = True

        # board canonicalization
        board_in = getattr(req, "board", "") or ""
        board = "".join(board_in) if isinstance(board_in, (list, tuple)) else str(board_in).replace(" ", "")
        bmask = _make_board_mask(board) if "board_mask_52" in self.cont_cols else None

        # base state
        pot_bb = float(getattr(req, "pot_bb", 0.0) or 0.0)
        stack_bb = float(getattr(req, "eff_stack_bb", getattr(req, "stack_bb", 0.0)) or 0.0)

        # hand id if needed
        hid: Optional[int] = None
        if "hand_id" in self.x_cols:
            hh = getattr(req, "hero_hand", None)
            hid = _hand169_id_from_code(hh) if hh else None
            if hid is None:
                # allow caller to pass req.hand_id instead
                hid = int(getattr(req, "hand_id", 0) or 0)

        # size/facing for postflop modes
        size_frac = 0.0
        if self.mode == "facing":
            facing, frac = self._infer_facing(req, hero_is_ip=hero_is_ip)
            size_frac = float(frac) if (facing and frac is not None) else 0.0

        # cluster id if used
        cluster_id = self._cluster_id(board) if ("board_cluster_id" in self.x_cols) else None

        # ctx
        ctx = req.ctx
        # stakes
        sid = None
        if "stakes_id" in self.x_cols:
            raw = getattr(getattr(req, "stakes", None), "value", None)
            sid = str(raw if raw is not None else getattr(req, "stakes", "2"))

        # --- assemble row following sidecar schema ---
        row: Dict[str, Any] = {}
        for name in self.x_cols:
            if name == "hero_pos":
                row[name] = ("IP" if (street > 0 and hero_is_ip) else ("OOP" if street > 0 else hpos))
            elif name == "ip_pos":
                row[name] = "IP"
            elif name == "oop_pos":
                row[name] = "OOP"
            elif name == "ctx":
                row[name] = ctx
            elif name == "street":
                row[name] = str(street)
            elif name == "board_cluster_id":
                row[name] = int(cluster_id or 0)
            elif name == "stakes_id":
                row[name] = sid if sid is not None else "2"
            elif name == "hand_id":
                row[name] = int(hid or 0)
            elif name in ("villain_pos",):
                row[name] = vpos
            else:
                # pass-through if present (preflop x_cols include villain_pos/facing flags in your cfg)
                row[name] = req.raw.get(name) if hasattr(req, "raw") and isinstance(req.raw, dict) else str(getattr(req, name, ""))

        cont: Dict[str, Any] = {}
        for name in self.cont_cols:
            if name == "board_mask_52":
                cont[name] = bmask or [0.0] * 52
            elif name == "pot_bb":
                cont[name] = pot_bb
            elif name == "stack_bb":
                cont[name] = stack_bb
            elif name == "size_frac":
                cont[name] = size_frac
            elif name == "faced_frac":
                cont[name] = float(getattr(req, "faced_size_frac", 0.0) or 0.0)
            else:
                cont[name] = float(getattr(req, name, 0.0) or 0.0)

        # --- encode & run ---
        x_cat = _encode_cats_one(self.x_cols, self.id_maps, row, self.device)
        x_cont = _encode_cont_one(self.cont_cols, cont, self.device)
        y = self.model(x_cat, x_cont)[0].detach().cpu().tolist()

        dbg = {
            "mode": self.mode,
            "ctx": ctx,
            "street": street,
            "cluster_id": int(cluster_id or 0) if cluster_id is not None else None,
            "size_frac": float(size_frac),
        }
        return EVOutput(actions=list(self.action_vocab), evs=[float(v) for v in y], debug=dbg)
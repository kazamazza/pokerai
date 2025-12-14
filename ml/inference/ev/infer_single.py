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
    available: bool
    actions: list[str]
    evs: list[float]
    evs_by_token: dict[str, float]
    debug: dict[str, Any] | None = None
    err: str | None = None


class EVInferSingle:
    """
    Sidecar-driven EV inferencer for exactly ONE split:
      mode ∈ {"preflop", "root", "facing"}.
    Produces EVs in sidecar action_vocab order (no routing, no token remapping).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        mode: str,  # "preflop" | "root" | "facing"
        x_cols: Sequence[str],
        cont_cols: Sequence[str],
        action_vocab: Sequence[str],
        id_maps: Dict[str, Dict[str, int]],
        device: Optional[torch.device] = None,
        clusterer: Optional[Any] = None,
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
        # fixed: avoid shadowing, normalize keys as str → int
        self.id_maps: Dict[str, Dict[str, int]] = {
            str(k): {str(a): int(b) for a, b in (mp or {}).items()}
            for k, mp in (id_maps or {}).items()
        }
        self.clusterer = clusterer

        # quick head-width sanity
        with torch.no_grad():
            x_cat = torch.zeros(1, len(self.x_cols), dtype=torch.long, device=self.device)
            cont_dim = sum(52 if c == "board_mask_52" else 1 for c in self.cont_cols)
            x_cont = torch.zeros(1, cont_dim, dtype=torch.float32, device=self.device)
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

        # sidecar schema
        x_cols = sc.get("cat_feature_order") or sc.get("x_cols") or sc.get("feature_order") or []
        cont_cols = sc.get("cont_feature_order") or sc.get("cont_cols") or []
        id_maps = sc.get("id_maps") or {}
        action_vocab = list(sc.get("action_vocab") or [])
        if not x_cols or not cont_cols or not action_vocab:
            missing = []
            if not x_cols:       missing.append("cat_feature_order/x_cols")
            if not cont_cols:    missing.append("cont_feature_order/cont_cols")
            if not action_vocab: missing.append("action_vocab")
            raise ValueError(f"incomplete EV sidecar: {sidecar_path} (missing: {', '.join(missing)})")

        # load wrapper; reconstruct net only for really old ckpts
        from ml.models.evnet import EVLit, EVNet, EVNetConfig
        try:
            lit = EVLit.load_from_checkpoint(str(checkpoint_path), map_location=dev)
            lit = lit.eval().to(dev)
        except TypeError:
            # back-compat path: rebuild EVNet from sidecar hints
            cat_cards = [len((id_maps or {}).get(c, {})) for c in x_cols]

            def _feat_size(n: str) -> int:
                return 52 if str(n) == "board_mask_52" else 1

            cont_dim = int(sum(_feat_size(n) for n in cont_cols))
            arch = sc.get("arch") or {}
            net = EVNet(EVNetConfig(
                cat_cardinalities=cat_cards,
                cont_dim=cont_dim,
                action_vocab=action_vocab,
                hidden_dims=arch.get("hidden_dims", [256, 256]),
                dropout=float(arch.get("dropout", 0.10)),
                max_emb_dim=int(arch.get("max_emb_dim", 32)),
            ))
            lit = EVLit.load_from_checkpoint(
                str(checkpoint_path), map_location=dev, net=net, lr=1e-3, weight_decay=1e-4
            ).eval().to(dev)

        return cls(
            model=lit,
            mode=str(mode),
            x_cols=list(x_cols),
            cont_cols=list(cont_cols),
            action_vocab=action_vocab,
            id_maps=id_maps,
            device=dev,
            clusterer=clusterer,
        )

    # ------- tiny helpers for row assembly -------
    def _hero_is_ip(self, street: int, hpos: str, vpos: str) -> bool:
        if street == 0:
            return False
        if hpos == "BTN" and vpos in ("SB", "BB"):
            return True
        if {hpos, vpos} == {"SB", "BB"}:
            return hpos == "BB"
        order = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
        try:
            return order.index(hpos) > order.index(vpos)
        except ValueError:
            return True

    def _ctx_guard(self, ctx: str) -> str:
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

    # --------------- SINGLE-MODE INFERENCE ---------------
    @torch.no_grad()
    def predict(self, req, *, tokens: Sequence[str] | None = None) -> EVOutput:
        # 1) assemble one row from req based on self.mode
        row_cat: dict[str, Any] = {}
        row_cont: dict[str, Any] = {}

        street = int(getattr(req, "street", 0) or 0)
        board = (getattr(req, "board", "") or "").replace(" ", "")
        pot_bb = float(getattr(req, "pot_bb", 0.0) or 0.0)
        stack = float(getattr(req, "eff_stack_bb", getattr(req, "stack_bb", 0.0)) or 0.0)

        # optional features
        if "board_mask_52" in self.cont_cols:
            row_cont["board_mask_52"] = _make_board_mask(board)
        if "board_cluster_id" in self.x_cols:
            cid = 0
            if self.clusterer:
                try:
                    cid = int(self.clusterer.predict(board))
                except Exception:
                    cid = 0
            row_cat["board_cluster_id"] = cid

        # common cont
        if "pot_bb" in self.cont_cols:   row_cont["pot_bb"] = pot_bb
        if "stack_bb" in self.cont_cols: row_cont["stack_bb"] = stack

        # ---- per-mode stuff ----
        if self.mode == "preflop":
            # ---- preflop sized-facing features ----
            facing = bool(getattr(req, "facing_bet", False))
            frac = float(getattr(req, "faced_size_frac", 0.0) or 0.0)
            free_check = (not facing) and (str(getattr(req, "hero_pos", "")).upper() == "BB")

            if "facing_flag" in self.x_cols: row_cat["facing_flag"] = int(facing)
            if "free_check" in self.x_cols: row_cat["free_check"] = int(free_check)
            if "faced_frac" in self.cont_cols: row_cont["faced_frac"] = frac

        elif self.mode == "facing":
            # require faced size
            frac = getattr(req, "faced_size_frac", None)
            if frac is None and getattr(req, "faced_size_pct", None) is not None:
                frac = float(getattr(req, "faced_size_pct")) / 100.0
            if frac is None:
                return EVOutput(False, [], [], {}, {"err": "missing_faced_size"}, "missing_faced_size")
            if "size_frac" in self.cont_cols: row_cont["size_frac"] = float(frac)

        elif self.mode == "root":
            if "size_frac" in self.cont_cols: row_cont["size_frac"] = 0.0

        # categorical pass-throughs
        def put(cat, val):
            if cat in self.x_cols: row_cat[cat] = val

        put("street", str(street))
        put("hero_pos", getattr(req, "hero_pos", ""))
        put("villain_pos", getattr(req, "villain_pos", ""))
        put("ip_pos", "IP");
        put("oop_pos", "OOP")  # sidecars use role tokens
        put("ctx", getattr(req, "ctx", None))
        put("stakes_id", str(getattr(req, "stakes", "2")))
        # hand id if needed
        if "hand_id" in self.x_cols:
            hid = getattr(req, "hand_id", None)
            if hid is None and getattr(req, "hero_hand", None):
                hid = _hand169_id_from_code(getattr(req, "hero_hand"))
            row_cat["hand_id"] = int(hid or 0)

        # 2) encode & run
        x_cat = _encode_cats_one(self.x_cols, self.id_maps, row_cat, self.device)
        x_cont = _encode_cont_one(self.cont_cols, row_cont, self.device)
        y = self.model(x_cat, x_cont)[0].detach().cpu().tolist()

        base_actions = list(self.action_vocab)
        base_map = {a: float(v) for a, v in zip(base_actions, y)}

        # 3) optional mapping to caller’s menu
        if tokens is not None:
            tok = [str(t) for t in tokens]
            mapped = [base_map.get(t, 0.0) for t in tok]
            return EVOutput(True, tok, mapped, {t: mapped[i] for i, t in enumerate(tok)},
                            {"mode": self.mode, "mapped_from": base_actions})

        return EVOutput(True, base_actions, [float(v) for v in y], base_map, {"mode": self.mode})
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn.functional as F

from ml.features.boards import BoardClusterer
from ml.inference.policy.types import PolicyResponse, PolicyRequest
from ml.inference.postflop_ctx import infer_postflop_ctx
from ml.models.policy_consts import ACTION_VOCAB as DEFAULT_VOCAB
from ml.models.postflop_policy_net import PostflopPolicyLit
from ml.utils.board_mask import make_board_mask_52
from ml.utils.sidecar import load_sidecar

DeviceLike = Union[str, torch.device]


def _to_device(dev: DeviceLike = "auto") -> torch.device:
    if isinstance(dev, torch.device):
        return dev
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


class PostflopPolicyInfer:
    """
    Inference wrapper for PostflopPolicyLit.
    - Loads a Lightning checkpoint (no nested .model).
    - Uses sidecar to align categorical feature order, cards (vocab sizes),
      optional id_maps, continuous features, and action vocab.
    - Provides batched predict_proba with per-side masks and temperature.
    """

    def __init__(
        self,
        *,
        model: PostflopPolicyLit,
        feature_order: Sequence[str],
        cards: Mapping[str, int],
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,
        cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
        action_vocab: Sequence[str] = None,
        device: Optional[torch.device] = None,
        clusterer: Optional[BoardClusterer] = None
    ):
        self.model = model.eval()
        self.device = device or _to_device("auto")
        self.model.to(self.device)
        self.feature_order = [str(c) for c in (feature_order or [])]
        if not self.feature_order:
            raise ValueError("Sidecar missing 'feature_order' (categorical feature order).")
        self.cards = {str(k): int(v) for k, v in (cards or {}).items()}
        self.clusterer = clusterer
        self.board_cluster_feat = "board_cluster" if "board_cluster" in self.feature_order \
            else ("board_cluster_id" if "board_cluster_id" in self.feature_order else None)
        missing = [c for c in self.feature_order if c not in self.cards]
        if missing:
            raise ValueError(f"Sidecar/cards missing entries for categorical features: {missing}")

        self.id_maps = {str(k): {str(a): int(b) for a, b in (m or {}).items()}
                        for k, m in (id_maps or {}).items()}
        self.cont_features = [str(c) for c in (cont_features or ["board_mask_52", "pot_bb", "eff_stack_bb"])]
        self.action_vocab = list(action_vocab) if action_vocab is not None else list(DEFAULT_VOCAB)
        self.vocab_size = len(self.action_vocab)

        # Sanity: verify head sizes match vocab
        self._assert_model_heads_match_vocab()

    def _legal_mask(self, *, facing_bet: bool, actor: str, bet_menu: Optional[Sequence[float]]) -> torch.Tensor:
        """
        Recreate the dataset’s legality rules, producing [V] mask on the same device.
        """
        names = self.action_vocab
        legal = set()
        BET_BUCKETS = {"BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100", "DONK_33"}
        RAISE_BUCKETS = {"RAISE_150", "RAISE_200", "RAISE_300", "RAISE_400", "RAISE_500", "ALLIN"}
        if facing_bet:
            legal = {"FOLD", "CALL"} | RAISE_BUCKETS
        else:
            legal = {"CHECK"} | BET_BUCKETS
            # DONK only OOP
            if actor.lower() != "oop" and "DONK_33" in legal:
                legal.remove("DONK_33")
            if bet_menu:
                want = set()
                # normalize menu → buckets
                if 0.25 in bet_menu: want.add("BET_25")
                if 0.33 in bet_menu: want.add("BET_33"); want.add("DONK_33")
                if 0.50 in bet_menu: want.add("BET_50")
                if 0.66 in bet_menu: want.add("BET_66")
                if 0.75 in bet_menu: want.add("BET_75")
                if 1.00 in bet_menu: want.add("BET_100")
                for b in {"BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100"}:
                    if b not in want and b in legal:
                        legal.remove(b)
                if "DONK_33" not in want and "DONK_33" in legal:
                    legal.remove("DONK_33")

        m = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
        for i, a in enumerate(names):
            if a in legal:
                m[i] = 1.0
        return m

    def _build_row_dict(self, req: "PolicyRequest") -> Dict[str, Any]:
        """
        Convert PolicyRequest → model row dict with all features present
        (categoricals in raw tokens, cont as floats; cluster id if available).
        No reliance on req.ip_pos/oop_pos (derived here).
        """
        # --- street (clamp to flop/turn/river for postflop) ---
        try:
            street = int(req.street or 1)
        except Exception:
            street = 1
        street = 1 if street < 1 else (3 if street > 3 else street)

        # --- seats: derive ip/oop from hero/villain ---
        hero = (req.hero_pos or "").strip().upper()
        vill = (req.villain_pos or "").strip().upper()

        # default safe fallbacks if missing
        if not hero and vill:
            hero = "BTN" if vill != "BTN" else "CO"
        if not vill and hero:
            vill = "BB" if hero != "BB" else "SB"
        if not hero and not vill:
            hero, vill = "BTN", "BB"

        try:
            hero_is_ip = PolicyRequest.is_hero_ip(hero, vill)
        except Exception:
            hero_is_ip = True

        ip_pos = hero if hero_is_ip else vill
        oop_pos = vill if hero_is_ip else hero

        # --- context (robust inference + explicit override via raw.ctx) ---
        ctx = infer_postflop_ctx(req)

        row: Dict[str, Any] = {
            "hero_pos": hero,
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "ctx": ctx,
            "street": street,
        }

        # --- board → mask + optional cluster ---
        if "board_mask_52" in getattr(self, "cont_features", []) and req.board:
            row["board_mask_52"] = make_board_mask_52(req.board)

        bcf = getattr(self, "board_cluster_feat", None)  # "board_cluster" or "board_cluster_id" or None
        if bcf:
            cid = 0
            if getattr(self, "clusterer", None) and req.board:
                try:
                    cid = int(self.clusterer.predict_one(req.board))
                except Exception:
                    cid = 0
            row[bcf] = cid

        # --- continuous scalars ---
        row["pot_bb"] = float(req.pot_bb or 0.0)
        row["eff_stack_bb"] = float(req.eff_stack_bb or 0.0)

        # --- optional bet menu passthrough (if present in raw) ---
        if isinstance(req.raw, dict) and "bet_sizes" in req.raw:
            try:
                row["bet_sizes"] = list(req.raw["bet_sizes"])
            except Exception:
                pass

        return row

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: Union[str, Path],
            sidecar_path: Union[str, Path],
            device: DeviceLike = "auto",
    ) -> "PostflopPolicyInfer":
        """
        Load a Lightning checkpoint + sidecar.
        Be tolerant to either schema:
          - feature_order + cards
          - cat_feature_order + card_sizes
        We do NOT modify the global load_sidecar(); we just fall back locally.
        """
        dev = _to_device(device)

        try:
            sc_raw = load_sidecar(sidecar_path)  # may raise if keys differ
        except Exception:
            import json
            p = Path(sidecar_path)
            if not p.exists():
                raise FileNotFoundError(f"Sidecar missing: {p}")
            try:
                sc_raw = json.loads(p.read_text())
            except Exception as e:
                raise ValueError(f"Failed to parse sidecar JSON at {p}: {e}")

        feature_order = sc_raw.get("feature_order") or sc_raw.get("cat_feature_order") or []
        cards_raw = sc_raw.get("cards") or sc_raw.get("card_sizes") or {}

        if not feature_order or not isinstance(feature_order, list):
            raise ValueError(
                f"Sidecar {sidecar_path} missing 'feature_order'/'cat_feature_order' list"
            )
        if not isinstance(cards_raw, dict) or not cards_raw:
            raise ValueError(
                f"Sidecar {sidecar_path} missing 'cards'/'card_sizes' dict"
            )

        cards = {str(k): int(v) for k, v in cards_raw.items()}
        id_maps = sc_raw.get("id_maps") or {}
        cont_features = sc_raw.get("cont_features") or ["board_mask_52", "pot_bb", "eff_stack_bb"]

        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as DEFAULT_VOCAB
        except Exception:
            DEFAULT_VOCAB = None
        action_vocab = sc_raw.get("action_vocab") or DEFAULT_VOCAB

        lit = PostflopPolicyLit.load_from_checkpoint(str(checkpoint_path), map_location=dev)
        lit.eval().to(dev)

        return cls(
            model=lit,
            feature_order=feature_order,
            cards=cards,
            id_maps=id_maps,
            cont_features=cont_features,
            action_vocab=action_vocab,
            device=dev,
        )

    @classmethod
    def from_dir(
        cls,
        ckpt_dir: Union[str, Path],
        ckpt_name: Optional[str] = None,
        device: DeviceLike = "auto",
    ) -> "PostflopPolicyInfer":
        """Convenience: load {dir}/best-or-last.ckpt + {dir}/sidecar.json"""
        ckpt_dir = Path(ckpt_dir)
        if ckpt_name is None:
            # Try best (pattern), else last.ckpt
            cands = sorted(ckpt_dir.glob("postflop_policy-*-*.ckpt"))
            checkpoint_path = cands[0] if cands else ckpt_dir / "best.ckpt"
        else:
            checkpoint_path = ckpt_dir / ckpt_name
        sidecar_path = ckpt_dir / "best_sidecar.json"
        return cls.from_checkpoint(checkpoint_path, sidecar_path, device=device)

    def _assert_model_heads_match_vocab(self) -> None:
        """Quick forward with zeros to verify head widths == action vocab."""
        # minimal fake batch B=1
        x_cat = {k: torch.zeros(1, dtype=torch.long, device=self.device) for k in self.feature_order}
        brd = torch.zeros(1, 52, device=self.device)
        x_cont = {
            "board_mask_52": brd,
            "pot_bb": torch.zeros(1, 1, device=self.device),
            "eff_stack_bb": torch.zeros(1, 1, device=self.device),
        }
        li, lo = self.model(x_cat, x_cont)
        if li.shape[-1] != self.vocab_size or lo.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Model head width != action_vocab size: "
                f"ip={li.shape[-1]}, oop={lo.shape[-1]}, vocab={self.vocab_size}"
            )

    def _encode_cat_value(self, col: str, v: Any) -> int:
        mapping = self.id_maps.get(col) or {}
        if mapping:
            key = "__NA__" if v is None else str(v)
            if key in mapping:
                return mapping[key]
            # Map unknowns to the reserved last id if available
            vocab = int(self.cards.get(col, 0))
            return vocab - 1 if vocab > 0 else 0
        # If no mapping present, fall back to int-cast (assumes caller already encoded)
        try:
            return int(v) if v is not None else 0
        except Exception:
            return 0

    def _encode_x_cat(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, List[int]] = {c: [] for c in self.feature_order}

        need_cluster = ("board_cluster_id" in self.feature_order)
        for r in rows:
            # If the model expects board_cluster_id but the row doesn't have it,
            # try to compute it from 'board' via cluster_fn if provided.
            r_local = dict(r)
            if need_cluster and "board_cluster_id" not in r_local:
                if self.cluster_fn and "board" in r_local and r_local["board"]:
                    try:
                        cid = int(self.cluster_fn(str(r_local["board"])))
                        r_local["board_cluster_id"] = cid
                    except Exception:
                        # leave missing; _encode_cat_value will map to unknown bucket
                        pass

            for c in self.feature_order:
                if c not in r_local:
                    raise KeyError(f"Missing categorical feature '{c}' in row keys {list(r.keys())}")
                out[c].append(self._encode_cat_value(c, r_local[c]))

        return {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in out.items()}

    def _as_52_mask(self, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            t = v.to(dtype=torch.float32, device=self.device).view(-1)
        else:
            import numpy as np
            arr = np.asarray(v, dtype=np.float32)  # force float32
            t = torch.tensor(arr, dtype=torch.float32, device=self.device).view(-1)
        if t.numel() < 52:
            pad = torch.zeros(52, dtype=torch.float32, device=self.device)
            pad[: t.numel()] = t
            t = pad
        elif t.numel() > 52:
            t = t[:52]
        return t

    def _encode_x_cont(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(rows)
        x_cont: Dict[str, torch.Tensor] = {}
        if "board_mask_52" in self.cont_features:
            masks = [self._as_52_mask(r.get("board_mask_52", [0] * 52)) for r in rows]
            x_cont["board_mask_52"] = torch.stack(masks, dim=0)  # already float32/device
        for k in ("pot_bb", "eff_stack_bb"):
            if k in self.cont_features:
                vals = [float(r.get(k, 0.0) or 0.0) for r in rows]
                x_cont[k] = torch.tensor(vals, dtype=torch.float32, device=self.device).view(B, 1)
        return x_cont

    def _role_masks(
        self,
        actor: str,
        B: int,
        mask_ip: Optional[torch.Tensor],
        mask_oop: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _norm(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if m is None:
                return None
            m = m.to(self.device).float()
            if m.dim() == 1:  # [V] -> [B,V]
                m = m.view(1, -1).repeat(B, 1)
            return m

        mi = _norm(mask_ip)
        mo = _norm(mask_oop)

        if mi is None or mo is None:
            ones = torch.ones(B, self.vocab_size, dtype=torch.float32, device=self.device)
            zeros = torch.zeros_like(ones)
            if (actor or "ip").lower() == "ip":
                mi = mi if mi is not None else ones
                mo = mo if mo is not None else zeros
            else:
                mi = mi if mi is not None else zeros
                mo = mo if mo is not None else ones
        return mi, mo

    @torch.no_grad()
    def predict_proba(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        actor: str = "ip",
        mask_ip: Optional[torch.Tensor] = None,   # [V] or [B,V]
        mask_oop: Optional[torch.Tensor] = None,  # [V] or [B,V]
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict of:
          - probs_ip  : [B,V]
          - probs_oop : [B,V]
          - (optional) logits_ip/logits_oop if return_logits=True
        """
        if not rows:
            B = 0
            empty = torch.empty(B, self.vocab_size, device=self.device)
            out = {"probs_ip": empty, "probs_oop": empty}
            if return_logits:
                out["logits_ip"], out["logits_oop"] = empty, empty
            return out

        B = len(rows)
        x_cat = self._encode_x_cat(rows)
        x_cont = self._encode_x_cont(rows)

        li, lo = self.model(x_cat, x_cont)  # [B,V], [B,V]
        if li.shape[-1] != self.vocab_size or lo.shape[-1] != self.vocab_size:
            raise RuntimeError(f"Model head width changed at runtime: ip={li.shape[-1]}, oop={lo.shape[-1]}, vocab={self.vocab_size}")

        mi, mo = self._role_masks(actor, B, mask_ip, mask_oop)

        big_neg = torch.finfo(li.dtype).min / 4
        li_masked = torch.where(mi > 0.5, li, big_neg)
        lo_masked = torch.where(mo > 0.5, lo, big_neg)

        if temperature and temperature != 1.0:
            t = float(temperature)
            li_masked = li_masked / t
            lo_masked = lo_masked / t

        probs_ip = F.softmax(li_masked, dim=-1)
        probs_oop = F.softmax(lo_masked, dim=-1)

        out: Dict[str, torch.Tensor] = {"probs_ip": probs_ip, "probs_oop": probs_oop}
        if return_logits:
            out["logits_ip"], out["logits_oop"] = li, lo
        return out


    @torch.no_grad()
    def predict(self,
                req: PolicyRequest,
                *,
                actor: str = "ip",
                temperature: float = 1.0) -> "PolicyResponse":
        """
        High-level: takes PolicyRequest, resolves board→mask/cluster, builds legal mask from
        actor & menu, and returns a PolicyResponse (actions/probs for the active side).
        """
        row = self._build_row_dict(req)

        # encode cats/conts for a batch of 1
        x_cat = self._encode_x_cat([row])
        x_cont = self._encode_x_cont([row])

        # legal mask (vector [V]); expand to [1,V]
        bet_menu = None
        if isinstance(row.get("bet_sizes"), (list, tuple)):
            bet_menu = [float(x) for x in row["bet_sizes"]]
        facing_bet = bool(req.facing_bet)
        m_side = self._legal_mask(facing_bet=facing_bet, actor=actor, bet_menu=bet_menu).view(1, -1)
        li, lo = self.model(x_cat, x_cont)
        logits = li if actor.lower() == "ip" else lo
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(m_side > 0.5, logits, big_neg)
        if temperature and temperature != 1.0:
            masked = masked / float(temperature)

        probs = torch.softmax(masked, dim=-1)[0]  # [V]
        actions = list(self.action_vocab)
        probs_np = [float(p) for p in probs.tolist()]

        return PolicyResponse(
            actions=actions,
            probs=probs_np,
            evs=[0.0] * len(actions),
            notes=[f"postflop policy; actor={actor}, temp={temperature}"],
            debug={
                "input_row": row,
                "actor": actor,
                "facing_bet": facing_bet,
                "mask_nonzero": int(m_side.sum().item()),
            },
        )
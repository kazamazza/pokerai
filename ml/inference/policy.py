# policy_infer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math

from ml.features.boards import BoardClusterer
from ml.features.hands import hand_to_169_label
from ml.inference.equity import EquityNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInfer
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import RangeNetPreflopInfer
from ml.inference.request.exploit_request import ExploitRequest

ACTION_VOCAB = []  # your real vocab will be imported in runtime path


@dataclass
class PolicyInferDeps:
    # optional
    pop: PopulationNetInfer | None = None
    # required
    exploit: ExploitNetInfer | None = None
    equity: EquityNetInfer | None = None
    # rangenets
    range_pre: RangeNetPreflopInfer | None = None
    # postflop policy
    policy_post: PostflopPolicyInfer | None = None
    # utils
    clusterer: BoardClusterer | None = None
    params: Dict[str, Any] | None = None


# --- add near imports ---
import math

class PolicyInfer:
    def __init__(self, deps: PolicyInferDeps):
        if deps.exploit is None:
            raise ValueError("exploit infer is required")
        if deps.equity is None:
            raise ValueError("equity infer is required")
        if deps.range_pre is None:
            raise ValueError("range_pre (RangeNetPreflopInfer) is required")

        # postflop policy can be stubbed (see below)
        self.pol_post = deps.policy_post

        self.pop   = deps.pop
        self.expl  = deps.exploit
        self.eq    = deps.equity
        self.rng_pre  = deps.range_pre
        self.clusterer = deps.clusterer
        self.p = deps.params or {}

        # Cache vocab once (and allow fallback for stub)
        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC
            self.action_vocab = list(_VOC)
        except Exception:
            # Safe default for stub/testing; adjust to your real vocab as needed
            self.action_vocab = ["CHECK", "BET_33", "BET_66", "FOLD", "CALL", "RAISE_200", "ALLIN"]

    # ---------- public ----------
    # ---------- public ----------
    def predict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Router:
          street==0  → preflop
          street in {1,2,3} → postflop (flop/turn/river)
        If street is missing/invalid, we try a light best-effort inference and default to preflop.
        """
        # Defensive copy so we can add inferred fields without mutating caller’s dict
        r = dict(req)

        # Normalize street
        try:
            street = int(r.get("street", 0))
        except Exception:
            street = 0

        # If street missing/odd, try to infer from board string length
        # (e.g., "AsKd7c" → flop; "AsKd7c2h" → turn; "AsKd7c2h8d" → river)
        if street not in (0, 1, 2, 3):
            board = str(r.get("board", "")).strip()
            n = len(board)
            if n >= 10:  # 5 cards
                street = 3
            elif n >= 8:  # 4 cards
                street = 2
            elif n >= 6:  # 3 cards
                street = 1
            else:
                street = 0
            r["street"] = street

        # Route
        if street == 0:
            return self._predict_preflop(r)
        else:
            # Flop/Turn/River share the same postflop path
            return self._predict_postflop(r)

    def _predict_preflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preflop flow:
          1) Get villain range via RangeNetPreflopInfer.
          2) (Optional) compute hero hand's prior weight within that range for debugging.
          3) Return a simple action stub so you can smoke-test end to end.

        NOTE: This is intentionally conservative: we don’t invent bet sizes or EV math here.
              It’s a scaffold you can replace with your real preflop policy when ready.
        """
        # 1) Build the preflop row for RangeNet
        row = {
            "stack_bb": float(req.get("stack_bb", req.get("eff_stack_bb", 100))),
            "hero_pos": str(req.get("hero_pos", "")).upper(),
            "opener_pos": str(req.get("opener_pos", "")).upper(),
            "opener_action": str(req.get("opener_action", "RAISE")).upper(),
            "ctx": str(req.get("ctx", "SRP")).upper(),
        }

        # Call RangeNet
        rng = self.rng_pre.predict_proba([row])  # torch.Tensor [1,169]
        if hasattr(rng, "detach"):
            rng = rng.detach().cpu().numpy()
        rng_169 = rng[0].tolist() if len(rng) else [1.0 / 169.0] * 169

        # 2) Optional: report hero hand’s prior mass in villain range (nice debug)
        hero_hand = str(req.get("hero_hand", "")).strip()
        hero_mass = None
        if hero_hand:
            label = hand_to_169_label(hero_hand)
            idx = self.rng_pre.hand_to_id.get(label) if hasattr(self.rng_pre, "hand_to_id") else None
            if idx is not None and 0 <= idx < 169:
                hero_mass = float(rng_169[idx])

        # 3) Minimal legal actions stub (so UI has something to display preflop)
        # If we're facing an open (BTN raised and we are BB), prefer CALL/RAISE/FOLD split.
        facing_open = bool(req.get("facing_open", False)) or (
                row["opener_action"] == "RAISE" and row["hero_pos"] in ("BB", "SB")
        )
        if facing_open:
            actions = ["FOLD", "CALL", "RAISE_300"]  # 3bb raise multiple as placeholder
            probs = [0.35, 0.40, 0.25]
        else:
            # opening or limp spots – keep it simple
            actions = ["FOLD", "RAISE_250"]  # 2.5x open placeholder
            probs = [0.30, 0.70]

        # Apply legality mask anyway
        actions, probs = self._apply_legality_mask(actions, probs, req)

        # Equity (optional) for telemetry
        eq = self._equity(req)

        return {
            "actions": actions,
            "probs": probs,
            "evs": [0.0] * len(actions),  # EV modeling for preflop can be wired later
            "debug": {
                "street": 0,
                "villain_range_169": rng_169,
                "hero_prior_mass_in_villain_range": hero_mass,
                "equity": eq,
            },
            "notes": ["preflop stub; range provided for downstream policy"],
        }

    def _exploit(self, req: Dict[str, Any]) -> Dict[str, float]:
        """
        Call ExploitInference and return a response mix in policy-space.
        Returns dict with at least 'fold','call','raise' in [0,1].
        Falls back to neutral if self.exploit is missing or request incomplete.
        """
        if getattr(self, "exploit", None) is None:
            # neutral fallback
            return {"fold": 1 / 3, "call": 1 / 3, "raise": 1 / 3}

        try:
            # Build an ExploitRequest expected by your ETL
            er = self._to_exploit_request(req)
            out = self.exploit.predict(er)  # {"p_fold","p_call","p_raise",...}
            return {
                "fold": float(out.get("p_fold", 1 / 3)),
                "call": float(out.get("p_call", 1 / 3)),
                "raise": float(out.get("p_raise", 1 / 3)),
            }
        except Exception:
            # be bulletproof: never let exploit failure crash the policy
            return {"fold": 1 / 3, "call": 1 / 3, "raise": 1 / 3}

    def _to_exploit_request(self, req: Dict[str, Any]) -> "ExploitRequest":
        """
        Minimal, robust adapter from policy-layer request -> ExploitRequest.
        It tries to reuse fields if present; otherwise builds safe defaults.
        """
        # --- seats ---
        seats = req.get("seats")
        if not seats or not isinstance(seats, list):
            # build a tiny 3-handed table as a fallback
            seats = [
                {"player_id": "p1", "name": "Alice", "actor": "Alice", "stack_size": float(req.get("stack_bb", 100.0)),
                 "pos_name": "BTN"},
                {"player_id": "p2", "name": "Bob", "actor": "Bob", "stack_size": float(req.get("stack_bb", 100.0)),
                 "pos_name": "SB"},
                {"player_id": "p3", "name": "Eve", "actor": "Eve", "stack_size": float(req.get("stack_bb", 100.0)),
                 "pos_name": "BB"},
            ]

        # --- actions (assume preflop if absent) ---
        actions = req.get("actions")
        if not actions or not isinstance(actions, list):
            actions = [{"actor": seats[0]["actor"], "act": 2, "amount_bb": 3.0, "street": int(req.get("street", 0))}]

        # --- street ---
        street = int(req.get("street", 0))  # 0..3

        # --- villain actor ---
        villain = req.get("villain_actor")
        if not villain:
            # pick the last action’s next player as villain if possible; else SB
            villain = seats[1]["actor"] if len(seats) > 1 else seats[0]["actor"]

        # --- stakes info ---
        bb = float(req.get("bb", req.get("stakes", {}).get("bb", 1.0)))
        stakes_info = {"bb": bb, "sb": bb / 2.0, "ante_bb": float(req.get("ante_bb", 0.0)), "id": 2}

        # --- position_by_player ---
        pos_by = req.get("position_by_player")
        if not isinstance(pos_by, dict):
            pos_by = {}
            for s in seats:
                pos_label = s.get("pos_name") or s.get("position") or "UTG"
                pos_by[s["actor"]] = {"name": pos_label, "player_id": s.get("player_id")}

        return ExploitRequest(
            street=street,
            villain_actor=villain,
            stake=req.get("stake", "nl10"),
            stakes_info=stakes_info,
            seats=seats,
            actions=actions,
            position_by_player=pos_by,
            player_rates=req.get("player_rates"),
            global_rates=req.get("global_rates"),
        )

    # =========================
    # 2) Postflop policy row builder
    # =========================
    def _postflop_policy_row(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the single feature-row the postflop policy model expects.
        If the model exposes 'feature_order', we fill those keys; otherwise return a sane generic row.
        """
        # Generic features available in most setups:
        street = int(req.get("street", 0))
        pot_bb = float(req.get("pot_bb", 0.0))
        eff_stack_bb = float(req.get("eff_stack_bb", req.get("stack_bb", 100.0)))
        # optional position label for villain (policy models often need it)
        villain_pos = None
        pos_by = req.get("position_by_player") or {}
        v = req.get("villain_actor")
        if isinstance(pos_by, dict) and v in pos_by:
            villain_pos = pos_by[v].get("name")
        villain_pos = villain_pos or req.get("villain_pos") or "BTN"

        row_default = {
            "street": street,
            "pot_bb": pot_bb,
            "eff_stack_bb": eff_stack_bb,
            "villain_pos": villain_pos,
            # add more if your policy model needs them:
            # "spr": eff_stack_bb / max(pot_bb, 1e-6),
            # "action_hist": some encoding, etc.
        }

        model = getattr(self, "pol_post", None)
        # If the policy model declares a feature order / schema, respect it
        feat_order = getattr(model, "feature_order", None)
        if isinstance(feat_order, (list, tuple)) and feat_order:
            out = {}
            for k in feat_order:
                out[k] = row_default.get(k)
            return out

        return row_default

    # ---------- postflop ----------
    def _predict_postflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Policy probs (stub or real)
        if self.pol_post is not None:
            row = self._postflop_policy_row(req)
            p_vec = self.pol_post.predict_proba([row])[0]  # np.ndarray [V]
            actions, probs = self._policy_vec_to_actions(p_vec)
        else:
            # ---- STUB for smoke testing: simple legal distribution
            actions = self.action_vocab
            probs = self._stub_probs(req, actions)

        # 1b) Legality mask (always)
        actions, probs = self._apply_legality_mask(actions, probs, req)

        # 2) Equity + response mix
        eq = self._equity(req)
        ex = self._exploit(req)
        pop = self._population(req)
        opp_mix = self._blend_response(ex, pop)

        # 3) EV per action (conservative)
        pot_bb   = float(req.get("pot_bb", 0.0))
        stack_bb = float(req.get("eff_stack_bb", req.get("stack_bb", 0.0)))
        evs = [self._ev_one(a, pot_bb, stack_bb, eq, opp_mix) for a in actions]

        # 4) Guardrails
        actions, probs, notes = self._guardrails(actions, probs, req, eq, {})

        return {
            "actions": actions,
            "probs": probs,
            "evs": evs,
            "debug": {
                "street": int(req["street"]),
                "equity": eq,
                "exploit": ex,
                "population": pop,
                "response_mix": opp_mix,
            },
            "notes": notes,
        }

    # ---------- equity / exploit / population ----------
    def _equity(self, req: Mapping[str, Any]) -> Dict[str, float]:
        """
        Use EquityNetInfer correctly:
          - Build a single row dict with the exact feature keys the model expects.
          - Preflop: no board_cluster_id; Postflop: include board_cluster_id if available.
        """
        street = int(req.get("street", 0))
        hero_hand = str(req.get("hero_hand", "")).strip()

        # Neutral prior if no known hero hand (allows smoke testing)
        if not hero_hand:
            return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}

        # Map to the hand_id/code you used in EquityNet (your infer encodes strings via id_maps)
        # For EquityNetInfer we just pass the raw "hand_id" integer if your sidecar encoders expect it,
        # but the safer universal route is to pass the *string label* and let the infer’s id_map handle it.
        # We used hand_id integers earlier, so keep consistent if your sidecar encoders are built that way.
        # If your EquityNet was trained with numeric hand_id, you must pass that numeric id.
        # Here we pass the label and let EquityNetInfer map it; if your encoders are numeric-only, replace with your id mapper.
        hand_label = hand_to_169_label(hero_hand)

        row: Dict[str, Any] = {"street": street, "hand_id": hand_label}

        if street > 0:
            # Try to attach board_cluster_id; if missing but clusterer exists, derive it
            cluster_id = req.get("board_cluster_id", None)
            if cluster_id is None and self.clusterer is not None:
                try:
                    board = str(req.get("board", ""))
                    if board:
                        cluster_id = int(self.clusterer.predict([board])[0])
                except Exception:
                    cluster_id = None
            if cluster_id is not None:
                row["board_cluster_id"] = int(cluster_id)

        # EquityNetInfer returns [[p_win, p_tie, p_lose]] from predict() or a tensor from predict_proba()
        out = self.eq.predict([row])  # -> List[List[float]]
        if not out:
            return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}
        p_win, p_tie, p_lose = out[0]
        return {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}

    def _blend_response(self, ex: Dict[str, float], pop: Optional[Dict[str, float]]) -> Dict[str, float]:
        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}
        w_raw = ex.get("weight", 0.0)
        try:
            w_raw = float(w_raw)
        except Exception:
            w_raw = 0.0
        w = max(0.0, min(1.0, w_raw / float(self.p.get("exploit_full_weight", 200))))
        return {
            "p_fold": (1 - w) * base["p_fold"] + w * ex.get("p_fold", base["p_fold"]),
            "p_call": (1 - w) * base["p_call"] + w * ex.get("p_call", base["p_call"]),
            "p_raise": (1 - w) * base["p_raise"] + w * ex.get("p_raise", base["p_raise"]),
        }

    # ---------- vocab & legality ----------
    def _policy_vec_to_actions(self, p_vec) -> tuple[list[str], list[float]]:
        if len(p_vec) != len(self.action_vocab):
            raise ValueError(f"Policy output mismatch: got {len(p_vec)} vs vocab {len(self.action_vocab)}")
        return self.action_vocab, list(map(float, p_vec))

    def _apply_legality_mask(self, actions: list[str], probs: list[float], req: Mapping[str, Any]) -> tuple[list[str], list[float]]:
        pairs = [(a, p) for a, p in zip(actions, probs) if self._is_legal(a, req)]
        if not pairs:
            # conservative fallback
            return ["FOLD"], [1.0]
        acts, ps = zip(*pairs)
        s = sum(ps)
        ps = [x / s for x in ps] if s > 0 else [1.0 / len(ps)] * len(ps)
        return list(acts), list(ps)

    def _is_legal(self, action: str, req: Mapping[str, Any]) -> bool:
        """
        Very light legality: disallow CHECK if facing bet; allow others.
        (You can enrich this later with stack/pot boundary checks.)
        """
        facing_bet = bool(req.get("facing_bet", False))
        up = str(action).upper()
        if facing_bet and up == "CHECK":
            return False
        return True

    # ---------- EV helpers (conservative) ----------
    def _ev_one(
        self,
        action: str,
        pot_bb: float,
        stack_bb: float,
        eq: Mapping[str, float],
        opp: Mapping[str, float],
    ) -> float:
        eq_win = float(eq.get("p_win", 0.5)); eq_tie = float(eq.get("p_tie", 0.0))
        e = self._safe(eq_win + 0.5 * eq_tie)
        p_fold = self._safe(opp.get("p_fold", 1/3)); p_call = self._safe(opp.get("p_call", 1/3)); p_raise = self._safe(opp.get("p_raise", 1/3))
        up = str(action).upper()

        if up in ("FOLD", "CHECK"):
            return 0.0

        if up == "CALL":
            # Without a facing-bet amount, avoid making up EV
            return 0.0

        invest = min(self._size_map(up, pot_bb, stack_bb), stack_bb)
        # Symmetric proxy: if called, villain matches invest; if fold, we win current pot
        final_pot = pot_bb + invest + invest
        ev_call  = e * final_pot - (1 - e) * invest
        return p_fold * pot_bb + (p_call + p_raise) * ev_call

    def _size_map(self, up: str, pot_bb: float, stack_bb: float) -> float:
        try:
            if up.startswith("BET_") or up.startswith("DONK_"):
                return float(up.split("_")[1]) / 100.0 * pot_bb
            if up.startswith("RAISE_"):
                mult = float(up.split("_")[1]) / 100.0
                return mult * pot_bb
            if up == "ALLIN":
                return max(0.0, float(stack_bb))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe(x: float, lo: float = 1e-9, hi: float = 1 - 1e-9) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(lo, min(hi, v))

    # ---------- postflop stub probs ----------
    def _stub_probs(self, req: Mapping[str, Any], actions: list[str]) -> list[float]:
        """
        Very simple prior:
          - If not facing bet: put mass on CHECK and BET_33.
          - If facing bet: put mass on FOLD/CALL/RAISE_200.
        """
        facing = bool(req.get("facing_bet", False))
        scores = []
        for a in actions:
            up = a.upper()
            if not facing:
                scores.append(0.5 if up == "CHECK" else (0.4 if up == "BET_33" else 0.1 if up == "BET_66" else 0.0))
            else:
                scores.append(0.45 if up == "CALL" else (0.4 if up in ("RAISE_200", "ALLIN") else 0.15 if up == "FOLD" else 0.0))
        s = sum(scores) or 1.0
        return [x / s for x in scores]
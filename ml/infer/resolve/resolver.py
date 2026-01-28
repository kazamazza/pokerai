from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Literal, cast
from ml.infer.types.observed_request import StackChangeEvent, PotChangeEvent, ObservedRequest
from ml.infer.types.resolved_state import ResolvedState, Topology, Ctx, Role, Street, NodeType


class ResolvedStateResolver:
    """
    PolicyRequest (observed) -> ResolvedState.

    v1 rules:
      - HU-only: exactly 2 distinct players in preflop street=0 stack_stream.
      - Infer ctx/topology/role + villain from street=0 stack_stream.
      - Infer ip/oop if we can infer villain_pos (seat_label) and hero_pos.
      - Infer node_type/faced_size_frac best-effort from current street streams.
    """

    _SPR_BINS: List[Tuple[float, float, str]] = [
        (0.0, 2.0, "SPR_0_2"),
        (2.0, 5.0, "SPR_2_5"),
        (5.0, 10.0, "SPR_5_10"),
        (10.0, float("inf"), "SPR_10_PLUS"),
    ]

    def resolve(self, req: ObservedRequest) -> ResolvedState:
        # allow_missing_villain=True because villain is inferred, not user-supplied
        self._validate_observed(req)

        hero_id = (req.hero_id or "").strip()
        if not hero_id:
            raise ValueError("Observed request must include hero_id (stable player id).")

        hero_pos = (req.hero_pos or "").strip().upper()
        if not hero_pos:
            raise ValueError("Observed request must include hero_pos (seat label).")

        street_i = int(req.street)
        if street_i not in (0, 1, 2, 3):
            raise ValueError(f"Illegal street={street_i}")
        street: Street = cast(Street, street_i)

        st = ResolvedState(
            stakes=str(req.stakes or "NL10").upper(),
            street=street,
            hero_id=hero_id,
            hero_pos=hero_pos,
            hero_hand=req.hero_hand,
            board=req.board,
            hand_id=req.hand_id,
            pot_bb=float(req.pot_bb or 0.0),
            eff_stack_bb=float(req.eff_stack_bb or 100.0),
            stack_stream=list(req.stack_stream or []),
            pot_stream=list(req.pot_stream or []),
            street_transitions=list(req.street_transitions or []),
            raw=dict(req.raw or {}),
        )

        # Postflop requires pot/stack
        if st.street in (1, 2, 3):
            if st.pot_bb <= 0.0 or st.eff_stack_bb <= 0.0:
                raise ValueError("Postflop requires pot_bb > 0 and eff_stack_bb > 0.")
            spr = st.eff_stack_bb / max(st.pot_bb, 1e-9)
            st = replace(st, spr=spr, spr_bin=self._spr_to_bin(spr))

        # ---- Infer ctx/topology/role + villain from preflop stream ----
        ctx, topo, role, villain_id, villain_pos, meta_ctx = self._infer_preflop_world(
            hero_id=st.hero_id,
            stack_stream=st.stack_stream,
        )

        st = replace(
            st,
            ctx=ctx,
            topology=topo,
            role=role,
            villain_id=villain_id,
            villain_pos=villain_pos,
            meta={**st.meta, **meta_ctx},
            reasons=st.reasons + ["ctx_from_preflop_stream"],
            confidence={**st.confidence, "ctx": 0.9, "villain": 0.9},
        )

        # ---- Infer ip/oop if we have both seats ----
        if st.villain_pos:
            ip_pos, oop_pos = self._infer_ip_oop(hero_pos=st.hero_pos, villain_pos=st.villain_pos, street=int(st.street))
            st = replace(st, ip_pos=ip_pos, oop_pos=oop_pos)

        # ---- Infer node_type + faced_size_frac (best-effort) ----
        node_type, faced_frac, meta_node = self._infer_node_type_and_faced(
            hero_id=st.hero_id,
            villain_id=st.villain_id,
            street=int(st.street),
            pot_bb=st.pot_bb,
            stack_stream=st.stack_stream,
            pot_stream=st.pot_stream,
        )

        st = replace(
            st,
            node_type=node_type,
            faced_size_frac=faced_frac,
            meta={**st.meta, **meta_node},
            confidence={**st.confidence, "node_type": 0.7},
        )

        return st

    # -------------------------
    # Preflop world inference
    # -------------------------
    def _infer_preflop_world(
        self,
        *,
        hero_id: str,
        stack_stream: List[StackChangeEvent],
    ) -> Tuple[Ctx, Topology, Role, Optional[str], Optional[str], Dict[str, Any]]:
        pre = [e for e in (stack_stream or []) if int(e.street) == 0]
        if not pre:
            raise ValueError("Need preflop stack_stream (street=0) to infer ctx/villain in v1.")

        # ✅ HU players = who appears in preflop stream (NOT invested>0)
        players = sorted({e.player_id for e in pre})
        if len(players) != 2:
            raise ValueError(f"v1 HU-only: expected 2 players in preflop stream, got {len(players)}: {players}")
        if hero_id not in players:
            raise ValueError(f"hero_id={hero_id} not in preflop stream players={players}")

        villain_id = players[0] if players[1] == hero_id else players[1]

        seat_of: Dict[str, str] = {}
        invested = defaultdict(float)  # pid -> total invested (sum of negative deltas)
        for e in sorted(pre, key=lambda x: x.tick):
            seat_of[e.player_id] = (e.seat_label or "").upper()
            invested[e.player_id] += max(0.0, -float(e.delta_bb))

        villain_pos = seat_of.get(villain_id)

        # Count “raises” as times someone becomes the new global max invested
        raises: List[Tuple[str, float]] = []
        running = defaultdict(float)
        global_max = 0.0

        for e in sorted(pre, key=lambda x: x.tick):
            pid = e.player_id
            put_in = max(0.0, -float(e.delta_bb))
            if put_in <= 0.0:
                continue
            running[pid] += put_in
            if running[pid] > global_max + 1e-6:
                global_max = running[pid]
                raises.append((pid, running[pid]))

        n_raises = len(raises)
        opener_id = raises[0][0] if n_raises >= 1 else None
        last_raiser_id = raises[-1][0] if raises else None

        max_invested = max(invested.values() or [0.0])
        opener_seat = seat_of.get(opener_id or "", "")
        villain_seat = seat_of.get(villain_id, "")

        meta: Dict[str, Any] = {
            "n_players_pre": len(players),
            "players_pre": players,
            "n_raises": n_raises,
            "opener_id": opener_id,
            "opener_seat": opener_seat,
            "last_raiser_id": last_raiser_id,
            "last_raiser_seat": seat_of.get(last_raiser_id or "", ""),
            "max_invested": float(max_invested),
            "hero_invested": float(invested.get(hero_id, 0.0)),
            "villain_invested": float(invested.get(villain_id, 0.0)),
        }

        # Limped single: nobody exceeds ~1bb invested (SB completes, BB checks)
        if n_raises == 0 or max_invested <= 1.0 + 1e-6:
            return "LIMPED_SINGLE", "LIMP", "ANY", villain_id, villain_pos, meta

        # Exactly 1 raise => VS_OPEN or BLIND_VS_STEAL
        if n_raises == 1:
            if opener_seat in {"CO", "BTN"} and villain_seat in {"SB", "BB"}:
                return "BLIND_VS_STEAL", "BVS", "ANY", villain_id, villain_pos, meta

            # Hero role from who opened
            role: Role = "AGGRESSOR" if opener_id == hero_id else "CALLER"
            return "VS_OPEN", "SRP", role, villain_id, villain_pos, meta

        # 2 raises => 3bet pot
        if n_raises == 2:
            role: Role = "AGGRESSOR" if last_raiser_id == hero_id else "CALLER"
            return "VS_3BET", "3BP", role, villain_id, villain_pos, meta

        # 3+ raises => treat as 4bet pot (cap)
        role = "AGGRESSOR" if last_raiser_id == hero_id else "CALLER"
        return "VS_4BET", "4BP", role, villain_id, villain_pos, meta

    # -------------------------
    # IP/OOP inference (seat order)
    # -------------------------
    def _infer_ip_oop(self, *, hero_pos: str, villain_pos: str, street: int) -> Tuple[Optional[str], Optional[str]]:
        order_pre = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        order_post = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        order = order_post if int(street) > 0 else order_pre

        h = hero_pos.upper()
        v = villain_pos.upper()
        if h not in order or v not in order:
            return None, None

        # IP acts later
        if order.index(h) > order.index(v):
            return h, v
        return v, h

    # -------------------------
    # Node type inference (best-effort)
    # -------------------------
    def _infer_node_type_and_faced(
        self,
        *,
        hero_id: str,
        villain_id: Optional[str],
        street: int,
        pot_bb: float,
        stack_stream: List[StackChangeEvent],
        pot_stream: List[PotChangeEvent],
    ) -> Tuple[NodeType, float, Dict[str, Any]]:
        if street == 0 or not villain_id:
            return "ROOT", 0.0, {"node_reason": "preflop_or_no_villain"}

        evs = sorted([e for e in (stack_stream or []) if int(e.street) == int(street)], key=lambda x: x.tick)
        if not evs:
            return "ROOT", 0.0, {"node_reason": "no_actions_current_street"}

        # last villain investment this street
        last_v: Optional[StackChangeEvent] = None
        for e in reversed(evs):
            if e.player_id == villain_id and (-float(e.delta_bb)) > 1e-6:
                last_v = e
                break
        if not last_v:
            return "ROOT", 0.0, {"node_reason": "no_villain_bet_current_street"}

        # if hero invested after that tick, then hero already responded
        for e in evs:
            if e.tick > last_v.tick and e.player_id == hero_id and (-float(e.delta_bb)) > 1e-6:
                return "ROOT", 0.0, {"node_reason": "hero_already_responded"}

        chips_in = max(0.0, -float(last_v.delta_bb))

        # try pot_before at same tick
        pot_before = None
        ps = sorted([p for p in (pot_stream or []) if int(p.street) == int(street)], key=lambda x: x.tick)
        for p in ps:
            if p.tick == last_v.tick:
                pot_before = float(p.pot_before_bb)
                break
        if pot_before is None:
            pot_before = float(pot_bb)

        faced_frac = float(chips_in) / max(float(pot_before), 1e-9)
        return "FACING", faced_frac, {
            "node_reason": "villain_bet_unanswered",
            "faced_chips_in": float(chips_in),
            "pot_before_for_frac": float(pot_before),
        }

    def _spr_to_bin(self, spr: float) -> str:
        x = float(spr)
        for lo, hi, label in self._SPR_BINS:
            if lo <= x < hi:
                return label
        return "SPR_UNKNOWN"

    def _validate_observed(self, req: ObservedRequest) -> None:
        if req.stakes is None or not str(req.stakes).strip():
            raise ValueError("ObservedRequest.stakes is required")

        if req.street not in (0, 1, 2, 3):
            raise ValueError(f"ObservedRequest.street must be 0..3, got {req.street}")

        if req.hero_pos is None or not str(req.hero_pos).strip():
            raise ValueError("ObservedRequest.hero_pos is required")

        # postflop requires pot/stack and board
        if req.street in (1, 2, 3):
            if req.pot_bb <= 0:
                raise ValueError("ObservedRequest.pot_bb must be > 0 for postflop")
            if req.eff_stack_bb <= 0:
                raise ValueError("ObservedRequest.eff_stack_bb must be > 0 for postflop")
            if not req.board:
                raise ValueError("ObservedRequest.board is required for postflop")

        # optional: normalize board formatting here if you want, but don’t mutate req
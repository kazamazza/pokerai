from ml.etl.exploit.exploit_nodes import VillainFeat, ExploitNode, NodeCtx, NodeLabel
from ml.etl.exploit.rolling_stats import RollingPreflopStats

ACT_TO_LABEL = {
    0: 0,  # fold
    1: 1,  # call
    2: 2,  # raise/open
    3: 1,  # check -> call-ish bucket (non-aggressive continue)
    4: 2,  # bet -> raise/aggressive bucket
}

import pandas as pd
from typing import Dict, Any, List

def _pos_of(player: str, pos_by_player: dict) -> str:
    # position_by_player: {"Aanakin57":{"id":2,"name":"CO"}, ...}
    return pos_by_player[player]["name"]

def _spr(stack_eff_bb: float, pot_bb: float) -> float:
    if pot_bb <= 0: return 100.0
    return float(min(100.0, max(0.0, stack_eff_bb / pot_bb)))

def _compact_seq(seq: List[Dict[str, Any]]) -> str:
    # human-readable history up to node
    parts = []
    for a in seq:
        amt = a.get("amount_bb")
        tag = f"{a['actor']}:{a['act']}" + ("" if amt is None else f"@{amt}")
        parts.append(tag)
    return ",".join(parts)

PREFLOP = 0

def update_tracker_from_hand(
    tracker: RollingPreflopStats,
    stake: str,
    hand: Dict[str, Any],
) -> None:
    """
    Call once per hand (after node emission) to update rolling stats.
    """
    seats_pids = [s["player_id"] for s in hand["seats"]]
    tracker.update_from_hand(
        stake=stake,
        seats_pids=seats_pids,
        hand_actions=hand["actions"],
    )

def _street_to_int(x):
    if isinstance(x, int):
        return x
    sx = str(x).lower()
    if sx in ("pre", "preflop", "0"): return 0
    if sx in ("flop", "1"): return 1
    if sx in ("turn", "2"): return 2
    if sx in ("river", "3"): return 3
    return 0  # fallback

def _act_to_code(a):
    # adapt to your schema; this is the mapping you used:
    # 0=fold, 1=call, 2=raise, 3=check, 4=bet
    return int(a)

def build_nodes_from_hand_json(
    h: Dict[str, Any],
    *,
    tracker: RollingPreflopStats,
) -> List[ExploitNode]:
    """
    Build ExploitNode rows from a single parsed hand.
    - Uses player_id (pid) for tracking rolling preflop stats (no stake keying).
    - Emits stakes_id from the hand into NodeCtx for downstream features.
    """
    nodes: List[ExploitNode] = []

    hand_id = h["hand_id"]
    stakes_id = h["stakes"]["id"]

    # position_by_player: {actor_name: {"id": <pos_id>, "name": "CO", "player_id": <pid>}}
    pos_by_actor: Dict[str, Dict[str, Any]] = h["position_by_player"]

    # Map actor -> pid and actor -> starting stack (BB)
    actor_to_pid: Dict[str, str] = {}
    seats_by_pid = {s["player_id"]: float(s["stack_size"]) for s in h["seats"]}
    seats_by_actor: Dict[str, float] = {}
    for actor, info in pos_by_actor.items():
        pid = info.get("player_id") or info.get("id")
        actor_to_pid[actor] = pid
        seats_by_actor[actor] = float(seats_by_pid.get(pid, 0.0))

    # ---------- PASS 1: per-hand preflop flags (pid-based) ----------
    PREFLOP = 0
    actions = h["actions"]
    preflop_actions = [a for a in actions if _street_to_int(a["street"]) == PREFLOP]

    vpip_flag: Dict[str, int] = {pid: 0 for pid in seats_by_pid}
    pfr_flag:  Dict[str, int] = {pid: 0 for pid in seats_by_pid}
    tb_flag:   Dict[str, int] = {pid: 0 for pid in seats_by_pid}
    f3b_flag:  Dict[str, int] = {pid: 0 for pid in seats_by_pid}

    raises_seen = 0
    for a in preflop_actions:
        actor = a["actor"]
        pid   = actor_to_pid.get(actor)
        if not pid:
            continue
        act = _act_to_code(a["act"])
        amt = float(a.get("amount_bb") or 0.0)

        if act in (1, 2) and amt > 0.0:
            vpip_flag[pid] = 1

        if act == 2 and amt > 0.0:
            if raises_seen == 0:
                pfr_flag[pid] = 1
            else:
                tb_flag[pid] = 1
            raises_seen += 1

    if raises_seen >= 2:
        seen_3bet = False
        for a in preflop_actions:
            actor = a["actor"]
            pid   = actor_to_pid.get(actor)
            if not pid:
                continue
            act = _act_to_code(a["act"])
            if act == 2 and not seen_3bet:
                seen_3bet = True
            elif seen_3bet and act == 0:
                f3b_flag[pid] = 1

    # ---------- PASS 2: walk actions and build nodes (attach rolling rates by pid) ----------
    committed_by_actor = {actor: 0.0 for actor in seats_by_actor}
    pot_bb = 0.0
    actions_so_far: List[Dict[str, Any]] = []

    for a in actions:
        actor = a["actor"]
        info = pos_by_actor.get(actor)
        pid  = actor_to_pid.get(actor)
        if not info or not pid:
            actions_so_far.append(a)
            continue

        act    = _act_to_code(a["act"])
        street = _street_to_int(a["street"])
        amt    = a.get("amount_bb")

        pot_before = pot_bb
        if amt is not None:
            committed_by_actor[actor] += float(amt)
            pot_bb += float(amt)

        if act not in ACT_TO_LABEL:
            actions_so_far.append(a)
            continue
        y = ACT_TO_LABEL[act]

        villain_pos = _pos_of(actor, pos_by_actor)  # e.g., "BTN","SB",...

        # effective stack at decision (rough)
        actor_start = seats_by_actor.get(actor, 0.0)
        actor_rem = max(0.0, actor_start - committed_by_actor[actor])
        others_rem = [
            max(0.0, seats_by_actor.get(other, 0.0) - committed_by_actor.get(other, 0.0))
            for other in seats_by_actor
            if other != actor
        ]
        eff_stack = min([actor_rem] + others_rem) if others_rem else actor_rem

        # rolling rates BEFORE counting this hand’s flags
        rates = tracker.get_rates(pid)  # expects pid only
        hands_obs = tracker.get_hands_observed(pid)

        vill = VillainFeat(
            hands_observed=hands_obs,
            vpip=float(rates.get("vpip", 0.0)),
            pfr=float(rates.get("pfr", 0.0)),
            three_bet=float(rates.get("three_bet", 0.0)),
            fold_to_3b=float(rates.get("fold_to_3b", 0.0)),
            wwsf=None,
            agg_factor=None,
        )

        ctx = NodeCtx(
            hand_id=hand_id,
            street=street,
            stakes_id=stakes_id,
            hero_pos=None,
            villain_pos=villain_pos,
            spr=_spr(eff_stack, pot_before),
            pot_bb=pot_before,
            action_seq=_compact_seq(actions_so_far),
        )

        nodes.append(ExploitNode(ctx=ctx, vill=vill, label=NodeLabel(action=y)))
        actions_so_far.append(a)

    # ---------- PASS 3: update tracker AFTER the hand (pid-based) ----------
    for actor, info in pos_by_actor.items():
        pid = actor_to_pid.get(actor)
        if not pid:
            continue
        tracker.observe_event(
            pid=pid,
            vpip=bool(vpip_flag.get(pid, 0)),
            pfr=bool(pfr_flag.get(pid, 0)),
            three_bet=bool(tb_flag.get(pid, 0)),
            fold_to_3b=bool(f3b_flag.get(pid, 0)),
        )

    return nodes

def flatten_nodes(nodes: List[ExploitNode]) -> pd.DataFrame:
    rows = []
    for n in nodes:
        rows.append({
            "hand_id": n.ctx.hand_id,
            "street": n.ctx.street,
            "stakes_id": n.ctx.stakes_id,
            "villain_pos": n.ctx.villain_pos,
            "spr": n.ctx.spr,
            "pot_bb": n.ctx.pot_bb,
            "action_seq": n.ctx.action_seq,
            "hands_obs": n.vill.hands_observed,
            "vpip": n.vill.vpip,
            "pfr": n.vill.pfr,
            "three_bet": n.vill.three_bet,
            "fold_to_3b": n.vill.fold_to_3b,
            "wwsf": n.vill.wwsf,
            "agg_factor": n.vill.agg_factor,
            "y_action": n.label.action,  # 0/1/2
            "w": 1.0,
        })
    return pd.DataFrame(rows)
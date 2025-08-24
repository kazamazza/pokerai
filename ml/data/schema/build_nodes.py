from ml.etl.exploit.exploit_nodes import VillainFeat, ExploitNode, NodeCtx, NodeLabel

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

def build_nodes_from_hand_json(h: Dict[str, Any]) -> List[ExploitNode]:
    nodes: List[ExploitNode] = []

    hand_id = h["hand_id"]
    stakes_id = h["stakes"]["id"]
    pos_by_player = h["position_by_player"]   # actor -> {"id":..,"name":..}
    seats = {s["player_id"]: float(s["stack_size"]) for s in h["seats"]}

    # crude running pot & commitments (BB units)
    # start from total blinds if you track them elsewhere; here we start from 0
    committed = {pid: 0.0 for pid in seats}
    pot_bb = 0.0

    actions_so_far: List[Dict[str, Any]] = []
    for a in h["actions"]:
        actor = a["actor"]
        act   = int(a["act"])
        amt   = a.get("amount_bb")  # can be None (check/fold)
        street = int(a["street"])

        # pot before this action:
        pot_before = pot_bb

        # update pot/commitments with this action for *next* node:
        if amt is not None:
            committed[actor] += float(amt)
            pot_bb += float(amt)

        # build node (what villain did here)
        if actor not in pos_by_player:  # safety
            actions_so_far.append(a)
            continue

        villain_pos = _pos_of(actor, pos_by_player)
        # effective stack in BB (remaining) — simple approximation
        # remaining for actor:
        actor_start = seats.get(actor, 0.0)
        actor_rem = max(0.0, actor_start - committed[actor])
        # crude effective vs table = min over others’ remaining
        others_rem = [max(0.0, seats.get(p,0.0) - committed.get(p,0.0)) for p in seats if p != actor]
        eff_stack = min([actor_rem] + others_rem) if others_rem else actor_rem

        # label
        if act not in ACT_TO_LABEL:
            actions_so_far.append(a);
            continue
        y = ACT_TO_LABEL[act]

        ctx = NodeCtx(
            hand_id=hand_id,
            street=street,
            stakes_id=stakes_id,
            hero_pos=None,                   # not required here
            villain_pos=villain_pos,
            spr=_spr(eff_stack, pot_before), # SPR at decision point
            pot_bb=pot_before,
            action_seq=_compact_seq(actions_so_far),
        )

        # villain features – if you have per-player stats, plug them here; else neutral
        vill = VillainFeat(
            hands_observed=0,
            vpip=0.0, pfr=0.0, three_bet=0.0, fold_to_3b=0.0,
            wwsf=None, agg_factor=None
        )

        nodes.append(ExploitNode(ctx=ctx, vill=vill, label=NodeLabel(action=y)))
        actions_so_far.append(a)

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
# ml/etl/population/derive_features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import polars as pl

# --- act ids (adapt if yours differ) ---
ACT_FOLD  = 0
ACT_CALL  = 1
ACT_RAISE = 2  # preflop raise or any raise
ACT_CHECK = 3
ACT_BET   = 4
ACT_ALLIN = 5  # will normalize to raise

STREET_MAP = {0:0, 1:1, 2:2, 3:3}  # raw->id

def _street_len(street_id: int) -> int:
    return {0:0, 1:3, 2:4, 3:5}[int(street_id)]

def _board_prefix(board_list: List[str], street_id: int) -> str:
    want = _street_len(street_id)
    if want == 0 or not board_list:
        return ""
    return "".join(board_list[:want])

@dataclass
class ActionRow:
    hand_id: str
    street_id: int
    seq_in_street: int
    actor_id: str
    actor_pos_id: int
    act_id: int
    amount_bb: float | None
    pot_before_bb: float
    to_call_bb: float
    facing_id: int
    bet_pct_of_pot: float
    spr_approx: float
    spr_bin: int
    board_cluster_id: int

def _spr_bin(x: float) -> int:
    if x < 2: return 0
    if x < 4: return 1
    if x < 7: return 2
    return 3

def _is_bet_like(act_id: int) -> bool:
    return act_id in (ACT_BET, ACT_RAISE, ACT_ALLIN)

def _is_call(act_id: int) -> bool:
    return act_id == ACT_CALL

def _is_check(act_id: int) -> bool:
    return act_id == ACT_CHECK

def _is_fold(act_id: int) -> bool:
    return act_id == ACT_FOLD

def build_action_features_from_hands(
    hands_path: str | Path,
    *,
    clusterer=None,  # must expose predict(list[str])->list[int] or predict_one(str)->int
    default_eff_stack_bb: float = 100.0,
) -> pl.DataFrame:
    """
    Read hands.jsonl (or .jsonl.gz), reconstruct per-action features needed
    for PopulationNet aggregation. Returns a Polars DataFrame with:
      ['hand_id','street_id','seq_in_street','actor_pos_id','act_id',
       'pot_before_bb','to_call_bb','facing_id','bet_pct_of_pot',
       'spr_bin','board_cluster_id']
    """
    # load as rows (we’ll iterate in Python; polars isn’t great for this specific stateful scan)
    df = pl.read_ndjson(str(hands_path))
    rows = df.to_dicts()

    out: List[ActionRow] = []

    for H in rows:
        hand_id = str(H["hand_id"])
        seats = H.get("seats", [])
        pos_by_player = H.get("position_by_player", {})  # {player: {id:..., name:...}}
        board_list = H.get("board", []) or []            # ["As","Kh","2d","..."]
        actions = H.get("actions", []) or []

        # initial stacks map
        init_stack = {s["player_id"]: float(s.get("stack_size", default_eff_stack_bb)) for s in seats}
        # track committed and remaining stacks
        committed_total = {pid: 0.0 for pid in init_stack}
        remaining = {pid: float(init_stack[pid]) for pid in init_stack}

        # per-street state
        current_street = 0
        seq_in_street = -1
        pot_before = 0.0
        # per-street bet to call (highest commitment on street)
        street_commit = {pid: 0.0 for pid in init_stack}
        to_call_level = 0.0  # max(street_commit.values())

        def _advance_street(new_street: int):
            nonlocal current_street, seq_in_street, to_call_level, street_commit
            current_street = new_street
            seq_in_street = -1
            to_call_level = 0.0
            street_commit = {pid: 0.0 for pid in init_stack}

        for a in actions:
            st_raw = int(a.get("street", 0))
            if st_raw != current_street:
                _advance_street(st_raw)

            seq_in_street += 1
            actor = a.get("actor")
            act_id = int(a.get("act", -1))
            amt = a.get("amount_bb", None)
            amt = float(amt) if amt is not None else None

            # normalize ALLIN → RAISE
            if act_id == ACT_ALLIN:
                act_id = ACT_RAISE

            # resolve actor_pos_id
            # pos_by_player: {player: {"id": <pos_id>, "name": "BTN"}}
            apos = pos_by_player.get(actor, {})
            actor_pos_id = int(apos.get("id", -1))

            # board cluster id for this street
            board_str = _board_prefix(board_list, st_raw)
            if clusterer is not None:
                try:
                    cid = int(clusterer.predict_one(board_str))
                except Exception:
                    try:
                        cid = int(clusterer.predict([board_str])[0])
                    except Exception:
                        cid = 0
            else:
                cid = 0

            # compute to_call for actor at this moment
            # to_call_level is the highest per-street commitment so far
            prev_commit = street_commit.get(actor, 0.0)
            to_call = max(to_call_level - prev_commit, 0.0)

            # effective stack approx BEFORE acting (vs largest remaining opponent)
            # (crude but serviceable for bins)
            hero_rem = remaining.get(actor, default_eff_stack_bb)
            vill_rem = 0.0
            for pid, rem in remaining.items():
                if pid == actor:
                    continue
                # only consider opponents not all-in (rem > 0)
                vill_rem = max(vill_rem, rem)
            eff_before = min(hero_rem, vill_rem) if vill_rem > 0 else hero_rem

            spr_approx = (eff_before / max(pot_before, 1e-6)) if pot_before > 0 else 999.0
            spr_bin = _spr_bin(spr_approx)

            facing = 1 if to_call > 0.0 else 0
            bet_pct = (to_call / max(pot_before, 1e-6)) if facing else -1.0

            out.append(ActionRow(
                hand_id=hand_id,
                street_id=STREET_MAP.get(st_raw, st_raw),
                seq_in_street=seq_in_street,
                actor_id=actor,
                actor_pos_id=actor_pos_id,
                act_id=act_id,
                amount_bb=amt,
                pot_before_bb=pot_before,
                to_call_bb=to_call,
                facing_id=facing,
                bet_pct_of_pot=bet_pct,
                spr_approx=spr_approx,
                spr_bin=spr_bin,
                board_cluster_id=cid,
            ))

            # ---- update pot/commitments AFTER the action ----
            # street_commit tracks put-in *this street*; committed_total is across whole hand
            if _is_check(act_id) or _is_fold(act_id):
                # no chips added to pot on this action
                pass
            elif _is_call(act_id):
                put = to_call
                street_commit[actor] = to_call_level
                committed_total[actor] += put
                remaining[actor] = max(0.0, remaining[actor] - put)
                pot_before += put
            elif _is_bet_like(act_id):
                # for bet/raise we trust 'amt' as *gross* put-in on this action
                put = float(amt or 0.0)
                street_commit[actor] = street_commit.get(actor, 0.0) + put
                committed_total[actor] += put
                remaining[actor] = max(0.0, remaining[actor] - put)
                pot_before += put
                to_call_level = max(to_call_level, street_commit[actor])
            else:
                # unknown act: do nothing
                pass

    # -> Polars frame
    feat_df = pl.from_dicts([r.__dict__ for r in out])
    return feat_df.select([
        "hand_id","street_id","seq_in_street","actor_pos_id","act_id",
        "pot_before_bb","to_call_bb","facing_id",
        pl.when(pl.col("facing_id") == 1)
        .then(
            (pl.col("to_call_bb") / pl.col("pot_before_bb").clip(lower_bound=1e-6))
            .clip(lower_bound=0.0, upper_bound=50.0)
        )
        .otherwise(pl.lit(-1.0))
        .alias("bet_pct_of_pot"),
        "spr_bin","board_cluster_id",
    ])
# ml/datasets/popnet_dataset.py
import os
from os import PathLike
from typing import Tuple, Dict, List, Union
import gzip, json, math
import torch
from torch.utils.data import Dataset

from ml.schema.pop_net_schema import PopNetSample

STREETS = ["preflop","flop","turn","river"]
POS6    = ["SB","BB","UTG","HJ","CO","BTN"]
ACTS    = ["fold","check","call","bet","raise","allin"]

# --- add this helper in population_net_dataset.py ---

def _expand_hand_to_decision_rows(hand: dict) -> list[dict]:
    """Convert one hand-level dict (with actions[]) into N decision-level {x,y} rows."""
    rows = []
    bb = float(hand.get("min_bet") or 0.1)
    positions = hand.get("position_by_player", {})
    pot_bb = float(hand.get("pot_bb") or 0.0)

    # precompute stacks in BB
    stacks_bb = {s["player_id"]: (float(s.get("stack_size") or 0.0) / bb)
                 for s in hand.get("seats", [])}

    last_bet_bb = None
    min_raise_to_bb = None
    street_counts = {"preflop": {"bet":0,"raise":0,"check":0,"call":0},
                     "flop":    {"bet":0,"raise":0,"check":0,"call":0},
                     "turn":    {"bet":0,"raise":0,"check":0,"call":0},
                     "river":   {"bet":0,"raise":0,"check":0,"call":0}}
    cur_street = "preflop"
    amount_to_call_bb_by_player = {}  # running per street

    for ev in hand.get("actions", []):
        # maintain street + simple pot geometry
        s = ev.get("street", cur_street) or cur_street
        if s != cur_street:
            cur_street = s
            last_bet_bb = None
            min_raise_to_bb = None
            amount_to_call_bb_by_player.clear()

        actor = ev.get("actor")
        act   = ev.get("act")
        amt   = ev.get("amount_bb")

        # update facing amounts (simple model: everyone faces last bet size)
        atc = float(amount_to_call_bb_by_player.get(actor, 0.0) or 0.0)

        # build (x,y)
        x = {
            "stake_tag": hand.get("stake_tag", hand.get("stakes","NL?")),
            "players": min(6, len(hand.get("seats", []))),
            "street": s,
            "actor": actor,
            "actor_pos": positions.get(actor, "UTG"),
            "btn_pos": positions.get(next((p for p,pos in positions.items() if pos=="BTN"), None), None),
            "positions": positions,
            "effective_stack_bb": min(stacks_bb.get(actor, 0.0),
                                      max([v for k,v in stacks_bb.items() if k != actor] + [0.0])),
            "pot_bb": float(pot_bb),
            "amount_to_call_bb": atc,
            "is_3bet_pot": False,         # can enrich later
            "is_4bet_plus": False,
            "is_first_in": False,
            "facing_open": atc > 0.0,
            "facing_3bet": False,
            "facing_4bet_plus": False,
            "board_cluster_id": None,
            "board_cards": hand.get("board") or None,
            "last_bet_bb": (None if last_bet_bb is None else float(last_bet_bb)),
            "min_raise_to_bb": (None if min_raise_to_bb is None else float(min_raise_to_bb)),
            "bets_this_street": street_counts[s]["bet"],
            "raises_this_street": street_counts[s]["raise"],
            "checks_this_street": street_counts[s]["check"],
            "calls_this_street": street_counts[s]["call"],
            "table_name": hand.get("table_name"),
            "stakes": hand.get("stakes"),
        }
        y = {
            "action": act,
            "amount_bb": (None if amt is None else float(amt)),
            "size_bucket": None
        }
        # accept only actions the schema allows (skip None / headers)
        if actor and act in {"fold","check","call","bet","raise","allin"}:
            rows.append({"x": x, "y": y})

        # update running state after acting
        if act in ("bet","raise","allin") and amt:
            last_bet_bb = float(amt)
            min_raise_to_bb = float(amt)  # naive; refine if you track increments
            pot_bb += float(amt)
            amount_to_call_bb_by_player = {}  # after a raise, reset “to call” map
        elif act == "call" and atc:
            pot_bb += float(atc)
            amount_to_call_bb_by_player[actor] = 0.0
        elif act == "check":
            pass
        elif act == "fold":
            amount_to_call_bb_by_player.pop(actor, None)

        if act in street_counts[s]:
            street_counts[s][act] += 1

    return rows

def one_hot(idx: int, n: int) -> List[float]:
    v = [0.0]*n
    if 0 <= idx < n: v[idx] = 1.0
    return v

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def safe(v, default=0.0):
    return float(v) if v is not None else float(default)

def scalarize_features(x) -> List[float]:
    """Turn PopNetFeatures into a flat numeric feature vector."""
    # one-hots
    street_oh = one_hot(STREETS.index(x.street), len(STREETS))
    pos_oh    = one_hot(POS6.index(x.actor_pos), len(POS6))

    # numeric scalars (log1p for stacks/pot; raw for amounts needed)
    eff = math.log1p(safe(x.effective_stack_bb, 0.0))
    pot = math.log1p(safe(x.pot_bb, 0.0))
    to_call = safe(x.amount_to_call_bb, 0.0)

    last_bet = safe(x.last_bet_bb, 0.0)
    min_raise_to = safe(x.min_raise_to_bb, 0.0)

    # flags
    flags = [
        float(bool(x.is_3bet_pot)),
        float(bool(x.is_4bet_plus)),
        float(bool(x.is_first_in)),
        float(bool(x.facing_open)),
        float(bool(x.facing_3bet)),
        float(bool(x.facing_4bet_plus)),
    ]

    # counts (clipped small range)
    counts = [
        float(clip(int(x.bets_this_street or 0),   0, 3)),
        float(clip(int(x.raises_this_street or 0), 0, 3)),
        float(clip(int(x.checks_this_street or 0), 0, 3)),
        float(clip(int(x.calls_this_street or 0),  0, 3)),
    ]

    return street_oh + pos_oh + [eff, pot, to_call, last_bet, min_raise_to] + flags + counts

def scalarize_label(y, pot_bb: float) -> List[float]:
    """7-dim target: 6-way one-hot action + 1 scalar size target."""
    act_idx = ACTS.index(y.action)
    action_oh = one_hot(act_idx, len(ACTS))
    pot = max(1.0, float(pot_bb or 0.0))
    if y.action in ("bet","raise","call","allin") and (y.amount_bb is not None):
        size_reg = clip(float(y.amount_bb)/pot, 0.0, 5.0)
    else:
        size_reg = 0.0
    return action_oh + [size_reg]

def open_fn(path, mode="rt", encoding="utf-8"):
    import gzip
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode, encoding=encoding)

class PopulationNetDataset(Dataset):
    def __init__(self, path: Union[str, os.PathLike, str]):
        self.rows: List[PopNetSample] = []

        with open_fn(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                # Accept decision-level rows directly, otherwise expand from hand-level
                if isinstance(raw, dict) and ("x" in raw and "y" in raw):
                    self.rows.append(PopNetSample(**raw))
                else:
                    for sample in _expand_hand_to_decision_rows(raw):
                        self.rows.append(PopNetSample(**sample))

        if not self.rows:
            raise ValueError(f"No usable rows in {path}")

        # Infer dims
        x0 = scalarize_features(self.rows[0].x)
        y0 = scalarize_label(self.rows[0].y, self.rows[0].x.pot_bb)
        self.input_dim  = len(x0)
        self.output_dim = len(y0)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        samp = self.rows[i]
        x_vec = torch.tensor(scalarize_features(samp.x), dtype=torch.float32)  # shape [D]
        y_vec = torch.tensor(scalarize_label(samp.y, samp.x.pot_bb), dtype=torch.float32)  # shape [M]
        return {"x_vec": x_vec}, y_vec
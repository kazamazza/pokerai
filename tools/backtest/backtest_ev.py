#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structural EV backtest:
- Reads your parsed hands.jsonl.gz
- Injects a hero seat (e.g., BTN or first-active)
- At each hero decision, queries your PolicyInfer (or a stub)
- Validates action legality given street state
- Computes EV for each legal candidate using equity from eval7
- Records whether chosen action is top-EV and aggregates stats

Install: pip install eval7
"""

from __future__ import annotations
import argparse, gzip, json, math, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import eval7  # fast hand/board equity calculator (PokerStove-like)

# =====================================================================================
# Constants / action names
# =====================================================================================

ACT_CHECK = "CHECK"
ACT_BET   = "BET"
ACT_CALL  = "CALL"
ACT_FOLD  = "FOLD"
ACT_RAISE = "RAISE"
ACT_ALLIN = "ALLIN"

STREET_PREFLOP = 0
STREET_FLOP    = 1
STREET_TURN    = 2
STREET_RIVER   = 3

# If you have a global ACTION_VOCAB for postflop policy labels, import it.
# Here we keep legality generation independent of label vocab.
DEFAULT_RAISE_PCTS = [150, 200, 300]      # raise-to as % of facing bet (fallback)
DEFAULT_BET_PCTS   = [33, 66, 100]        # bet as % of pot when not facing a bet

# =====================================================================================
# IO helpers
# =====================================================================================

def stream_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# =====================================================================================
# Policy adapter (wire your PolicyInfer here)
# =====================================================================================

class PolicyAdapter:
    """
    Thin adapter around your PolicyInfer (or a stub).
    Required method: decide(req) -> dict with {"act": str, "amount_bb": float|None}
    Optional: estimate_resp_probs(req) -> dict {p_fold,p_call,p_raise}
    """
    def __init__(self, mode: str = "REPLAY", inference: Any = None):
        self.mode = mode
        self.infer = inference  # plug your PolicyInfer here

    def decide(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.infer is not None:
            out = self.infer.predict(req)
            # Map to a simple {"act","amount_bb"}; if your infer already does this, return directly.
            actions: List[str] = out.get("actions", [])
            probs:   List[float] = out.get("probs", [])
            if actions and probs:
                i = max(range(len(probs)), key=lambda k: probs[k])
                return {"act": actions[i], "amount_bb": req.get("chosen_amount_bb")}
        # Fallback “match dataset if possible” (we supply dataset_act/amount in req for convenience)
        act = normalize_act(req.get("dataset_act"))
        amt = float(req.get("dataset_amount_bb") or 0.0) if act in (ACT_BET, ACT_RAISE) else None
        return {"act": act, "amount_bb": amt}

    def estimate_resp_probs(self, req: Dict[str, Any]) -> Dict[str, float]:
        """
        If you have Exploit/Population nets, use them here.
        This fallback is a mild, stationary guess.
        """
        # Simple defaults (tunable): a bit more folds at lower SPR, more calls at higher streets
        spr = safe_div(req.get("stack_bb", 0.0), max(1e-6, req.get("pot_bb", 1.0)))
        street = int(req.get("street", STREET_FLOP))
        base_fold = 0.35 if spr > 2.0 else 0.45
        base_call = 0.45 if street <= STREET_TURN else 0.40
        base_raise = 1.0 - base_fold - base_call
        return clamp_mix({"p_fold": base_fold, "p_call": base_call, "p_raise": max(0.05, base_raise)})

# =====================================================================================
# Light hand state + helpers
# =====================================================================================

@dataclass
class DecisionRecord:
    hand_id: str
    street: int
    pot_bb: float
    stack_bb: float
    facing_to_call_bb: float
    legal_actions: List[str]
    chosen_act: str
    chosen_amt: Optional[float]
    ev_by_action: Dict[str, float]
    chosen_is_top_ev: bool
    equity_vs_range: float

@dataclass
class BacktestEVStats:
    n_hands: int = 0
    n_decisions: int = 0
    n_legal: int = 0
    n_top_ev: int = 0
    ev_margin_avg: float = 0.0  # avg( EV(chosen) - EV(best) )
    records: List[DecisionRecord] = field(default_factory=list)

    def add(self, rec: DecisionRecord):
        self.n_decisions += 1
        self.n_legal += 1  # we only add if legal
        if rec.chosen_is_top_ev:
            self.n_top_ev += 1
        evs = list(rec.ev_by_action.values())
        if evs:
            ev_best = max(evs)
            ev_ch = rec.ev_by_action.get(rec.chosen_act, 0.0)
            self.ev_margin_avg += (ev_ch - ev_best)

    def finalize(self):
        if self.n_decisions:
            self.ev_margin_avg /= self.n_decisions

# =====================================================================================
# Core EV machinery (PokerStove-like via eval7)
# =====================================================================================

def card_str_to_eval7(card: str) -> eval7.Card:
    # Accept "Ah", "Td", etc.
    return eval7.Card(card)

def parse_board(board_list: List[str]) -> List[eval7.Card]:
    return [card_str_to_eval7(c) for c in board_list]

def hand_to_eval7_two(hero_hand: Sequence[str]) -> Tuple[eval7.Card, eval7.Card]:
    return card_str_to_eval7(hero_hand[0]), card_str_to_eval7(hero_hand[1])

def enumerate_villain_combos(range_hands: Sequence[str], dead: set[eval7.Card]) -> List[Tuple[eval7.Card, eval7.Card]]:
    """range_hands like ['AKs','QJo','77', ...] — expand to combos excluding dead cards."""
    out = []
    ranks = "23456789TJQKA"
    suits = "cdhs"
    def all_suited(r1, r2):
        pairs = []
        for s in suits:
            c1 = eval7.Card(r1 + s); c2 = eval7.Card(r2 + s)
            pairs.append((c1, c2))
        return pairs
    def all_offsuit(r1, r2):
        pairs = []
        for s1 in suits:
            for s2 in suits:
                if s1 == s2: continue
                c1 = eval7.Card(r1 + s1); c2 = eval7.Card(r2 + s2)
                pairs.append((c1, c2))
        return pairs
    def all_pair(r):
        pairs = []
        for i, s1 in enumerate(suits):
            for j, s2 in enumerate(suits):
                if j <= i: continue
                c1 = eval7.Card(r + s1); c2 = eval7.Card(r + s2)
                pairs.append((c1, c2))
        return pairs

    for spec in range_hands:
        spec = spec.strip().upper()
        if not spec: continue
        if len(spec) == 2 and spec[0] == spec[1]:  # pocket pair like '77'
            r = spec[0]
            for c1, c2 in all_pair(r):
                if c1 in dead or c2 in dead: continue
                out.append((c1, c2))
        elif len(spec) == 3:  # e.g., 'AKs' / 'QJo'
            r1, r2, tx = spec[0], spec[1], spec[2]
            if tx == 'S':
                for c1, c2 in all_suited(r1, r2):
                    if c1 in dead or c2 in dead: continue
                    out.append((c1, c2))
            elif tx == 'O':
                for c1, c2 in all_offsuit(r1, r2):
                    if c1 in dead or c2 in dead: continue
                    out.append((c1, c2))
        else:
            # fallback: treat as offsuit+suited union
            if len(spec) >= 2:
                r1, r2 = spec[0], spec[1]
                for c1, c2 in all_offsuit(r1, r2) + all_suited(r1, r2):
                    if c1 in dead or c2 in dead: continue
                    out.append((c1, c2))
    return out

def equity_vs_range(
    hero_cards: Tuple[eval7.Card, eval7.Card],
    board_cards: List[eval7.Card],
    villain_range_specs: Sequence[str],
    n_mc: int = 4000,
) -> float:
    """Estimate hero equity vs villain range on current board (Monte Carlo)."""
    dead = set(board_cards) | set(hero_cards)
    v_combos = enumerate_villain_combos(villain_range_specs, dead)
    if not v_combos:
        return 0.5  # neutral fallback

    deck = [c for c in eval7.Deck() if c not in dead]
    wins = ties = total = 0

    need = 5 - len(board_cards)  # cards left to draw
    for _ in range(n_mc):
        v1, v2 = random.choice(v_combos)
        if v1 in dead or v2 in dead:  # just in case
            continue
        # sample the rest of the board
        rnd = random.sample([c for c in deck if c not in (v1, v2)], need)
        full_board = board_cards + rnd
        hero_val = eval7.evaluate(list(hero_cards) + full_board)
        vill_val = eval7.evaluate([v1, v2] + full_board)
        if hero_val > vill_val:
            wins += 1
        elif hero_val == vill_val:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total

# =====================================================================================
# Legality & EV model
# =====================================================================================

def normalize_act(a: Any) -> str:
    u = str(a).upper()
    return ACT_RAISE if u == ACT_ALLIN else u

def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a) / float(b) if b else 0.0

def clamp_mix(m: Dict[str, float]) -> Dict[str, float]:
    a = max(1e-6, min(1.0 - 2e-6, float(m.get("p_fold", 0.33))))
    b = max(1e-6, min(1.0 - 1e-6 - a, float(m.get("p_call", 0.33))))
    c = max(1e-6, 1.0 - a - b)
    return {"p_fold": a, "p_call": b, "p_raise": c}

def last_bet_to_amount(events: List[Dict[str, Any]], up_to_idx: int) -> float:
    """Return the most recent 'to' amount in BB on the current street, else 0."""
    cur_street = events[up_to_idx]["street"]
    last_to = 0.0
    for j in range(up_to_idx, -1, -1):
        ev = events[j]
        if ev["street"] != cur_street:
            break
        act = normalize_act(ev["act"])
        if act in (ACT_BET, ACT_RAISE):
            last_to = float(ev.get("amount_bb", 0.0) or 0.0)
            break
    return last_to

def pot_before_index(events: List[Dict[str, Any]], up_to_idx: int) -> float:
    """Approximate pot BB before event index (coarse but sufficient for legality/EV sizing)."""
    pot = 0.0
    for j, ev in enumerate(events[:up_to_idx]):
        a = normalize_act(ev["act"])
        if a in (ACT_BET, ACT_RAISE, ACT_CALL):
            pot += float(ev.get("amount_bb", 0.0) or 0.0)
    return pot

def legal_actions(
    facing_to_call_bb: float,
    pot_bb: float,
    stack_bb: float,
    bet_pcts: Sequence[int] = DEFAULT_BET_PCTS,
    raise_pcts: Sequence[int] = DEFAULT_RAISE_PCTS,
) -> List[Tuple[str, Optional[float]]]:
    acts: List[Tuple[str, Optional[float]]] = []
    if facing_to_call_bb <= 1e-9:
        # Not facing a bet
        acts.append((ACT_CHECK, None))
        for p in bet_pcts:
            amt = max(0.01, (p / 100.0) * pot_bb)
            if amt < stack_bb:
                acts.append((ACT_BET, amt))
            else:
                acts.append((ACT_ALLIN, stack_bb))
                break
    else:
        # Facing a bet
        acts.append((ACT_FOLD, None))
        call_amt = min(stack_bb, facing_to_call_bb)
        acts.append((ACT_CALL, call_amt))
        for rp in raise_pcts:
            raise_to = max(call_amt * (rp / 100.0), call_amt * 2.0)
            if raise_to < stack_bb:
                acts.append((ACT_RAISE, raise_to))
            else:
                acts.append((ACT_ALLIN, stack_bb))
                break
    return acts

def ev_of_action(
    act: str,
    amt_bb: Optional[float],
    pot_bb: float,
    to_call_bb: float,
    stack_bb: float,
    eq: float,
    resp: Dict[str, float],
) -> float:
    """
    Simple one-step EV:
      - If FOLD/CHECK: 0 (incremental)
      - If CALL: evaluates final pot after call
      - If BET/RAISE: fold/call/raise mix is taken from resp
    """
    act = normalize_act(act)
    p_fold, p_call, p_raise = resp["p_fold"], resp["p_call"], resp["p_raise"]
    eq = float(eq)
    invest = float(amt_bb or 0.0)
    invest = min(invest, stack_bb)

    if act == ACT_CHECK:
        return 0.0
    if act == ACT_FOLD:
        return 0.0
    if act == ACT_CALL:
        call_amt = min(stack_bb, to_call_bb)
        final_pot = pot_bb + call_amt + call_amt
        return eq * final_pot - (1.0 - eq) * call_amt

    if act in (ACT_BET, ACT_RAISE, ACT_ALLIN):
        win_if_fold = pot_bb
        call_amt = invest
        final_pot = pot_bb + call_amt + call_amt
        ev_call  = eq * final_pot - (1.0 - eq) * call_amt
        # For simplicity we treat “raise back” like a call branch (conservative)
        ev_raise = ev_call
        return p_fold * win_if_fold + p_call * ev_call + p_raise * ev_raise

    return 0.0

# =====================================================================================
# Runner
# =====================================================================================

@dataclass
class RunnerConfig:
    hands_path: Path
    hero_policy: str = "BTN"            # "BTN","CO","SB","BB" or "first-active"
    max_hands: Optional[int] = None
    seed: int = 42
    # equity / EV settings
    mc_samples: int = 4000
    # output
    decisions_csv: Optional[Path] = None

def choose_hero(hand: Dict[str, Any], policy_label: str) -> Optional[str]:
    pos_by_player = hand.get("position_by_player") or {}
    if not pos_by_player:
        acts = hand.get("actions") or []
        return acts[0]["actor"] if acts else None

    if policy_label == "first-active":
        for ev in hand.get("actions") or []:
            if ev["actor"] in pos_by_player:
                return ev["actor"]
        return None

    target = policy_label.upper()
    for pid, pos in pos_by_player.items():
        if str(pos.get("name","")).upper() == target:
            return pid
    return None

def run_ev_backtest(cfg: RunnerConfig, policy: PolicyAdapter) -> BacktestEVStats:
    random.seed(cfg.seed)
    stats = BacktestEVStats()

    for i, hand in enumerate(stream_jsonl_gz(cfg.hands_path), start=1):
        if cfg.max_hands and i > cfg.max_hands:
            break
        stats.n_hands += 1

        hero = choose_hero(hand, cfg.hero_policy)
        if not hero:
            continue

        hero_cards = hand.get("hole_cards_by_player", {}).get(hero)
        if not hero_cards or len(hero_cards) != 2:
            # We need hero’s hand to compute equity; skip otherwise
            continue

        board_cards = parse_board(hand.get("board", []))
        events = hand.get("actions") or []

        # simple villain range proxy: use a broad “reasonable” range (replace with your preflop infer)
        villain_range_specs = default_villain_range_specs(hand)  # e.g., ["22","A2s","K9o",...]

        for idx, ev in enumerate(events):
            if ev.get("actor") != hero:
                continue

            street = int(ev.get("street", STREET_PREFLOP))
            pot_bb = pot_before_index(events, idx)
            to_call_bb = last_bet_to_amount(events, idx)  # “to” amount the hero faces if any
            stack_bb = max(1.0, float(hand.get("min_bet", 1.0)) * 100.0)  # coarse fallback if unknown
            # Legal action set at this node
            candidates = legal_actions(to_call_bb, pot_bb, stack_bb)

            # Equity vs villain range on current board
            h1, h2 = hand_to_eval7_two(hero_cards)
            eq = equity_vs_range((h1, h2), board_cards, villain_range_specs, n_mc=cfg.mc_samples)

            # Response mix (plug exploit/pop nets if you have them)
            req = {
                "street": street,
                "pot_bb": pot_bb,
                "stack_bb": stack_bb,
                "to_call_bb": to_call_bb,
                "hero_pos": (hand.get("position_by_player", {}).get(hero, {}).get("name") or ""),
                "dataset_act": ev.get("act"),
                "dataset_amount_bb": ev.get("amount_bb"),
            }
            resp = policy.estimate_resp_probs(req)

            # EV for each candidate
            ev_by_action: Dict[str, float] = {}
            for act, amt in candidates:
                ev_by_action[action_key(act, amt)] = ev_of_action(act, amt, pot_bb, to_call_bb, stack_bb, eq, resp)

            # Policy’s choice
            # We pass dataset act/amount for fallback matching; your real PolicyInfer can pick its own.
            chosen = policy.decide({
                **req,
                "chosen_amount_bb": ev.get("amount_bb"),
            })
            chosen_key = action_key(normalize_act(chosen["act"]), chosen.get("amount_bb"))

            # Verify legality (must exist among candidates)
            legal_keys = set(ev_by_action.keys())
            if chosen_key not in legal_keys:
                # Map illegal (e.g., CHECK facing bet) → mark illegal and continue
                continue

            # Was chosen action top-EV?
            top_ev = max(ev_by_action.values()) if ev_by_action else 0.0
            chosen_ev = ev_by_action.get(chosen_key, -1e9)
            chosen_is_top = abs(chosen_ev - top_ev) < 1e-9

            rec = DecisionRecord(
                hand_id=hand.get("hand_id",""),
                street=street,
                pot_bb=pot_bb,
                stack_bb=stack_bb,
                facing_to_call_bb=to_call_bb,
                legal_actions=sorted(list(legal_keys)),
                chosen_act=chosen_key,
                chosen_amt=chosen.get("amount_bb"),
                ev_by_action=ev_by_action,
                chosen_is_top_ev=chosen_is_top,
                equity_vs_range=eq,
            )
            stats.add(rec)
            if cfg.decisions_csv:
                append_decision_csv(cfg.decisions_csv, rec)

    stats.finalize()
    return stats

# =====================================================================================
# Glue & small utilities
# =====================================================================================

def default_villain_range_specs(hand: Dict[str, Any]) -> List[str]:
    """
    Very rough, static range (replace with your RangeNet/Exploit later).
    Keep wide so equity isn’t overly optimistic.
    """
    return [
        # Pairs
        "22","33","44","55","66","77","88","99","TT","JJ","QQ","KK","AA",
        # Suited big cards
        "A2s","A3s","A4s","A5s","A6s","A7s","A8s","A9s","ATs","AJs","AQs","AKs",
        "K9s","KTs","KJs","KQs","QTs","QJs","JTs","T9s","98s","87s","76s",
        # Offsuit broadways
        "A9o","ATo","AJo","AQo","AKo","KTo","KJo","KQo","QJo","JTo","T9o"
    ]

def action_key(act: str, amt: Optional[float]) -> str:
    if act in (ACT_BET, ACT_RAISE, ACT_ALLIN) and amt is not None:
        return f"{act}_{round(float(amt), 4)}"
    return act

def append_decision_csv(path: Path, rec: DecisionRecord):
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("hand_id,street,pot_bb,stack_bb,to_call_bb,chosen,top_ev,equity,legal,evs\n")
        evs_str = ";".join(f"{k}:{v:.4f}" for k, v in rec.ev_by_action.items())
        f.write(f"{rec.hand_id},{rec.street},{rec.pot_bb:.4f},{rec.stack_bb:.4f},{rec.facing_to_call_bb:.4f},"
                f"{rec.chosen_act},{int(rec.chosen_is_top_ev)},{rec.equity_vs_range:.4f},"
                f"\"{'|'.join(rec.legal_actions)}\",\"{evs_str}\"\n")

# =====================================================================================
# CLI
# =====================================================================================

def main():
    ap = argparse.ArgumentParser("Structural EV backtest (PokerStove-like via eval7)")
    ap.add_argument("--hands", required=True, type=Path, help="Path to hands.jsonl.gz")
    ap.add_argument("--hero", default="BTN", help='Hero seat: BTN/CO/SB/BB or "first-active"')
    ap.add_argument("--max-hands", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-samples", type=int, default=4000, help="Monte Carlo samples for equity (eval7)")
    ap.add_argument("--decisions-csv", type=Path, default=None, help="Write per-decision diagnostics")
    args = ap.parse_args()

    cfg = RunnerConfig(
        hands_path=args.hands,
        hero_policy=args.hero,
        max_hands=args.max_hands,
        seed=args.seed,
        mc_samples=args.mc_samples,
        decisions_csv=args.decisions_csv,
    )

    # Plug your real PolicyInfer here if you want (replace None).
    policy = PolicyAdapter(mode="EV_PLUG", inference=None)

    stats = run_ev_backtest(cfg, policy)

    print("\n=== STRUCTURAL EV BACKTEST ===")
    print(f"hands processed:            {stats.n_hands}")
    print(f"hero decisions evaluated:   {stats.n_decisions}")
    print(f"legal decisions:            {stats.n_legal}")
    print(f"top-EV chosen (%):          {100.0 * safe_div(stats.n_top_ev, max(1, stats.n_decisions)):.2f}%")
    print(f"avg EV margin (chosen-best):{stats.ev_margin_avg:.4f} BB")
    if cfg.decisions_csv:
        print(f"per-decision csv:           {cfg.decisions_csv}")

if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ml.core.contracts import ActionProb, PolicyRequest, SignalBundle, PolicyResponse


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    # clip negatives, renorm, fallback uniform
    keys = list(d.keys())
    vals = [max(0.0, float(d[k])) for k in keys]
    s = sum(vals)
    if s <= 0.0:
        u = 1.0 / max(1, len(keys))
        return {k: u for k in keys}
    return {k: v / s for k, v in zip(keys, vals)}


def _sorted_action_probs(d: dict[str, float]) -> list[ActionProb]:
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    return [ActionProb(action=a, p=float(p)) for a, p in items]


def _mix_logit_space(
    base: Dict[str, float],
    nudges: Dict[str, float],
    *,
    strength: float,
    cap: float,
) -> Dict[str, float]:
    """
    Robust blender:
      - convert base probs to log space
      - add bounded nudge deltas (interpreted as "preference" weights)
      - convert back and normalize
    """
    import math

    base_n = _normalize(base)
    keys = list(base_n.keys())

    # log(p) with floor to avoid -inf
    eps = 1e-9
    logits = {k: math.log(max(eps, base_n[k])) for k in keys}

    # apply bounded nudges
    for k, dv in nudges.items():
        if k not in logits:
            continue
        delta = max(-cap, min(cap, float(dv))) * float(strength)
        logits[k] += delta

    # softmax
    m = max(logits.values()) if logits else 0.0
    exps = {k: math.exp(logits[k] - m) for k in keys}
    s = sum(exps.values())
    if s <= 0.0:
        return base_n
    return {k: exps[k] / s for k in keys}


def _top_choice(d: Dict[str, float]) -> Tuple[str, float]:
    a, p = max(d.items(), key=lambda kv: kv[1])
    return a, float(p)


@dataclass
class SignalMixerConfig:
    # How hard non-GTO signals can push (keep conservative v1)
    ev_strength: float = 0.35
    equity_strength: float = 0.20
    exploit_strength: float = 0.45

    # Per-signal cap in logit units (prevents "go crazy" behaviour)
    nudge_cap: float = 1.25

    # If base is missing, what to do
    uniform_if_missing: bool = True


class SignalMixer:
    def __init__(self, cfg: Optional[SignalMixerConfig] = None) -> None:
        self.cfg = cfg or SignalMixerConfig()

    def mix(self, *, policy_request: PolicyRequest, signals: SignalBundle) -> PolicyResponse:
        """
        Decide final action distribution.

        Priority for base distribution:
          1) postflop GTO (root/facing model)
          2) preflop engine distribution (if street=0)
          3) fallback uniform over whatever actions exist in signals/legals
        """
        debug: Dict[str, Any] = {}

        # --- 1) pick base distribution ---
        base: Optional[Dict[str, float]] = None
        base_src = None

        if getattr(signals, "gto", None) and getattr(signals.gto, "action_probs", None):
            base = dict(signals.gto.action_probs)
            base_src = "gto"
        elif getattr(signals, "preflop", None) and getattr(signals.preflop, "action_probs", None):
            base = dict(signals.preflop.action_probs)
            base_src = "preflop"
        else:
            # try to infer any action set from other signals
            action_set = set()
            for key in ("ev", "equity", "exploit"):
                block = getattr(signals, key, None)
                ap = getattr(block, "action_probs", None) if block else None
                if ap:
                    action_set |= set(ap.keys())
            if action_set and self.cfg.uniform_if_missing:
                u = 1.0 / len(action_set)
                base = {a: u for a in sorted(action_set)}
                base_src = "uniform_from_signals"

        if not base:
            # absolute last resort: minimal poker actions (you can replace with legality later)
            action_set = ["fold", "check", "call", "bet", "raise"]
            u = 1.0 / len(action_set)
            base = {a: u for a in action_set}
            base_src = "uniform_fallback"

        base = _normalize(base)
        debug["base_src"] = base_src

        # --- 2) build nudges from signals ---
        # Each nudge dict is "delta preference per action" (positive pushes up).
        # Keep it simple: we expect upstream components to produce per-action nudges
        # or you can generate them from scalar metrics later.
        nudges_total: Dict[str, float] = {k: 0.0 for k in base.keys()}

        # EV nudge
        if getattr(signals, "ev", None) and getattr(signals.ev, "action_nudges", None):
            for a, dv in signals.ev.action_nudges.items():
                if a in nudges_total:
                    nudges_total[a] += float(dv) * self.cfg.ev_strength
            debug["used_ev"] = True
        else:
            debug["used_ev"] = False

        # Equity nudge
        if getattr(signals, "equity", None) and getattr(signals.equity, "action_nudges", None):
            for a, dv in signals.equity.action_nudges.items():
                if a in nudges_total:
                    nudges_total[a] += float(dv) * self.cfg.equity_strength
            debug["used_equity"] = True
        else:
            debug["used_equity"] = False

        # Exploit nudge
        if getattr(signals, "exploit", None) and getattr(signals.exploit, "action_nudges", None):
            for a, dv in signals.exploit.action_nudges.items():
                if a in nudges_total:
                    nudges_total[a] += float(dv) * self.cfg.exploit_strength
            debug["used_exploit"] = True
        else:
            debug["used_exploit"] = False

        # --- 3) apply nudges (bounded) ---
        # Note: nudges_total already has strengths applied; cap is absolute.
        final_probs = _mix_logit_space(
            base=base,
            nudges=nudges_total,
            strength=1.0,
            cap=self.cfg.nudge_cap,
        )

        # --- 4) build response ---
        top_a, top_p = _top_choice(final_probs)

        # quick human-ish summary (keep short)
        summary = f"Base={base_src}. Top={top_a} ({top_p:.2f})."

        return PolicyResponse(
            street=int(policy_request.street),
            node_type=policy_request.node_type,
            action_probs=_sorted_action_probs(final_probs),
            recommendation=top_a,
            summary=summary,
            signals=signals,
            debug=debug,
        )
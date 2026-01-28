# ml/infer/policy_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Any, Dict

from ml.core.contracts import PolicyRequest, SignalBundle, PolicyResponse
from ml.infer.mixer.mixer import SignalMixer
from ml.infer.resolve.resolver import ResolvedStateResolver
from ml.infer.policy_request_builder import PolicyRequestBuilder
from ml.infer.types.observed_request import ObservedRequest


# --- Optional protocols (helps you + mypy/pyright, but not required) ---

class PreflopEngine(Protocol):
    def predict(self, pr: PolicyRequest) -> SignalBundle: ...

class PostflopPredictor(Protocol):
    def predict_bundle(self, pr: PolicyRequest) -> SignalBundle: ...
    # If your predictor currently returns dict[action->prob], wrap it inside predict_bundle.

class ScalarEngine(Protocol):
    def predict(self, pr: PolicyRequest) -> SignalBundle: ...

@dataclass
class PolicyRuntime:
    resolver: ResolvedStateResolver
    builder: PolicyRequestBuilder
    mixer: SignalMixer

    # components (some may be None until trained)
    postflop_predictor: Optional[PostflopPredictor] = None
    preflop_engine: Optional[PreflopEngine] = None
    equity_engine: Optional[ScalarEngine] = None
    ev_engine: Optional[ScalarEngine] = None
    exploit_store: Optional[ScalarEngine] = None

    def infer(self, obs: ObservedRequest) -> PolicyResponse:
        st = self.resolver.resolve(obs)
        pr = self.builder.build(st)

        # 1) collect signals (keep each block isolated)
        bundle = SignalBundle.empty(street=pr.street, stakes_id=getattr(pr, "stakes_id", None))

        # preflop
        if pr.street == 0 and self.preflop_engine is not None:
            bundle = bundle.merge(self.preflop_engine.predict(pr))

        # postflop
        if pr.street in (1, 2, 3) and self.postflop_predictor is not None:
            bundle = bundle.merge(self.postflop_predictor.predict_bundle(pr))

        # later engines
        if self.equity_engine is not None:
            bundle = bundle.merge(self.equity_engine.predict(pr))

        if self.ev_engine is not None:
            bundle = bundle.merge(self.ev_engine.predict(pr))

        if self.exploit_store is not None:
            bundle = bundle.merge(self.exploit_store.predict(pr))

        # 2) mix → response
        return self.mixer.mix(policy_request=pr, signals=bundle)
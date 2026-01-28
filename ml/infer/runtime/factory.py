# ml/infer/factory.py
from __future__ import annotations

from ml.infer.mixer.mixer import SignalMixer
from ml.infer.resolve.resolver import ResolvedStateResolver
from ml.infer.policy_request_builder import PolicyRequestBuilder
from ml.infer.runtime.runtime import PolicyRuntime


class PolicyInferFactory:
    def create(self) -> PolicyRuntime:
        resolver = ResolvedStateResolver()
        builder = PolicyRequestBuilder()
        mixer = SignalMixer()

        # Load real deps later; start with None stubs
        postflop_predictor = None
        preflop_engine = None
        equity_engine = None
        ev_engine = None
        exploit_store = None

        return PolicyRuntime(
            resolver=resolver,
            builder=builder,
            mixer=mixer,
            postflop_predictor=postflop_predictor,
            preflop_engine=preflop_engine,
            equity_engine=equity_engine,
            ev_engine=ev_engine,
            exploit_store=exploit_store,
        )
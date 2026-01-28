from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ml.infer.runtime.factory import PolicyInferFactory
from ml.infer.runtime.runtime import PolicyRuntime

RUNTIME_KEY = "policy_runtime"

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: build runtime once
    runtime: PolicyRuntime = PolicyInferFactory().create()
    app.state.__setattr__(RUNTIME_KEY, runtime)
    yield
    # Shutdown: optional cleanup hooks
    rt: PolicyRuntime | None = getattr(app.state, RUNTIME_KEY, None)
    if rt and hasattr(rt, "close"):
        rt.close()

def set_runtime(app: FastAPI, rt: PolicyRuntime) -> None:
    setattr(app.state, RUNTIME_KEY, rt)

def get_runtime(app: FastAPI) -> PolicyRuntime:
    rt = getattr(app.state, RUNTIME_KEY, None)
    if rt is None:
        raise RuntimeError("PolicyRuntime not initialized. Did you call set_runtime(app, rt) on startup?")
    return rt
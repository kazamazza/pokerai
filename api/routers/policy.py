from fastapi import APIRouter, Depends, Request
from api.deps import get_runtime
from ml.core.contracts import PolicyResponse
from ml.infer.runtime.runtime import PolicyRuntime
from ml.infer.types.observed_request import ObservedRequest

router = APIRouter(tags=["policy"])

def runtime_dep(req: Request) -> PolicyRuntime:
    return get_runtime(req.app)

@router.post("/policy", response_model=PolicyResponse)
def infer_policy(obs: ObservedRequest, rt: PolicyRuntime = Depends(runtime_dep)) -> PolicyResponse:
    return rt.infer(obs)
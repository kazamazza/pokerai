from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"ok": True}

@router.get("/status")
def status():
    # expand later: model versions, commit hash, uptime, etc.
    return {"service": "policy", "status": "running"}
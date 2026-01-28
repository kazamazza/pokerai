from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.policy import router as policy_router
from api.deps import lifespan, set_runtime
from ml.infer.runtime.factory import PolicyInferFactory


def create_app() -> FastAPI:
    app = FastAPI(title="Poker Policy API", version="1.0.0", lifespan=lifespan)

    rt = PolicyInferFactory().create()
    set_runtime(app, rt)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/v1")
    app.include_router(policy_router, prefix="/v1")

    return app

app = create_app()
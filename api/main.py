from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml.inference.policy.policy_factory import PolicyInferFactory
from ml.inference.policy.types import PolicyRequest, PolicyResponse

app = FastAPI()

# Allow Postman or frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build full dependency tree once at startup
policy_engine = PolicyInferFactory().create()

@app.post("/policy", response_model=PolicyResponse)
def infer_policy(req: PolicyRequest):
    return policy_engine.predict(req)

@app.get("/ping")
def ping():
    return {"status": "ok"}
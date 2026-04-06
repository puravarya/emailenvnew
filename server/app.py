from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv

# Main app
app = FastAPI()

# API app (mounted under /api)
api = FastAPI()

env = EmailEnv()

class StepRequest(BaseModel):
    action: str

# -------------------------
# OpenEnv REQUIRED ENDPOINTS
# -------------------------

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/reset")
def reset():
    state = env.reset()
    return {
        "observation": state,
        "reward": 0,
        "done": False
    }

@api.post("/step")
def step(req: StepRequest):
    return env.step(req.action)

@api.get("/state")
def state():
    return {
        "observation": env.current_email
    }

# -------------------------
# Mount API under /api
# -------------------------
app.mount("/api", api)
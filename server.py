from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv

app = FastAPI()

env = EmailEnv()

# -------------------------
# Models
# -------------------------
class StepRequest(BaseModel):
    action: str


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Reset
# -------------------------
@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "observation": state,
        "reward": 0,
        "done": False
    }


# -------------------------
# Step
# -------------------------
@app.post("/step")
def step(req: StepRequest):
    result = env.step(req.action)

    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "done": result["done"]
    }


# -------------------------
# State
# -------------------------
@app.get("/state")
def state():
    return {
        "observation": env.current_email
    }
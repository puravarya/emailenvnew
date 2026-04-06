from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv

app = FastAPI()
api = FastAPI()

env = EmailEnv()
total_reward = 0  # 🔥 track total reward

class StepRequest(BaseModel):
    action: str

# -------------------------
# FUNCTIONS (shared logic)
# -------------------------

def reset_env():
    global total_reward
    total_reward = 0
    state = env.reset()
    return {
        "observation": state,
        "reward": 0,
        "total_reward": total_reward,
        "done": False
    }

def step_env(action: str):
    global total_reward
    result = env.step(action)
    total_reward += result["reward"]

    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "total_reward": total_reward,
        "done": result["done"]
    }

# -------------------------
# API ROUTES (/api)
# -------------------------

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/reset")
def reset_api():
    return reset_env()

@api.post("/step")
def step_api(req: StepRequest):
    return step_env(req.action)

@api.get("/state")
def state_api():
    return {
        "observation": env.current_email,
        "total_reward": total_reward
    }

# -------------------------
# DIRECT ROUTES (fallback)
# -------------------------

@app.get("/health")
def health_root():
    return {"status": "ok"}

@app.post("/reset")
def reset_root():
    return reset_env()

@app.post("/step")
def step_root(req: StepRequest):
    return step_env(req.action)

@app.get("/state")
def state_root():
    return {
        "observation": env.current_email,
        "total_reward": total_reward
    }

# -------------------------
# MOUNT API
# -------------------------
app.mount("/api", api)
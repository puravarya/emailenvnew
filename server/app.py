from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv

app = FastAPI(docs_url="/docs", redoc_url=None)

env = EmailEnv()
total_reward = 0  # track total reward


# -------------------------
# Request Model
# -------------------------
class StepRequest(BaseModel):
    action: str


# -------------------------
# Helper functions
# -------------------------
def normalize_action(action: str):
    return action.lower().replace("mark_", "").strip()


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

    action = normalize_action(action)  # 🔥 FIX

    result = env.step(action)
    total_reward += result["reward"]

    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "total_reward": total_reward,
        "done": result["done"]
    }


# -------------------------
# API ROUTES
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    return reset_env()


@app.post("/step")
def step(req: StepRequest):
    return step_env(req.action)


@app.get("/state")
def state():
    return {
        "observation": env.current_email,
        "total_reward": total_reward
    }
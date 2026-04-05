from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv

app = FastAPI()

env = EmailEnv()

class StepRequest(BaseModel):
    action: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "observation": state,
        "reward": 0,
        "done": False
    }

@app.post("/step")
def step(req: StepRequest):
    result = env.step(req.action)
    return result

@app.get("/state")
def state():
    return {"observation": env.current_email}
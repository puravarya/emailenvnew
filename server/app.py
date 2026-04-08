import sys
import os

# Ensure root-level modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv
import grader as grader_module

app = FastAPI(docs_url="/docs", redoc_url=None)

env = EmailEnv()
total_reward = 0.0


# -------------------------
# Request Model
# -------------------------
class StepRequest(BaseModel):
    action: str


# -------------------------
# Helpers
# -------------------------
def normalize_action(action: str):
    return action.lower().replace("mark_", "").strip()


def reset_env():
    global total_reward
    total_reward = 0.0
    state = env.reset()
    return {
        "observation": state,
        "reward": 0.0,
        "total_reward": total_reward,
        "done": False
    }


def step_env(action: str):
    global total_reward

    action = normalize_action(action)
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


# -------------------------
# TASKS ENDPOINT (REQUIRED)
# -------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "email_classification",
                "description": "Classify emails as spam, important, or social",
                "actions": ["spam", "important", "social"]
            },
            {
                "task_id": "spam_detection",
                "description": "Detect whether an email is spam or not_spam",
                "actions": ["spam", "not_spam"]
            },
            {
                "task_id": "email_priority",
                "description": "Assign priority (urgent, normal, low)",
                "actions": ["urgent", "normal", "low"]
            }
        ]
    }


# -------------------------
# GRADER ENDPOINT (CRITICAL FIX)
# -------------------------
@app.get("/grader")
def grader():
    return {
        "tasks": [
            {
                "task_id": "email_classification",
                "score": grader_module.grade_easy()
            },
            {
                "task_id": "spam_detection",
                "score": grader_module.grade_spam_easy()
            },
            {
                "task_id": "email_priority",
                "score": grader_module.grade_priority_easy()
            }
        ]
    }


# -------------------------
# MAIN ENTRY (OpenEnv requirement)
# -------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
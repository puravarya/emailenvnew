import sys
import os

# Ensure root-level modules (env.py, grader.py) are importable from server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv
import grader as grader_module

app = FastAPI(docs_url="/docs", redoc_url=None)

env = EmailEnv()
total_reward = 0.0


# -------------------------
# Request Models
# -------------------------
class StepRequest(BaseModel):
    action: str


class GraderRequest(BaseModel):
    task_id: str  # email_classification | spam_detection | email_priority


# -------------------------
# Helpers
# -------------------------
def normalize_action(action: str):
    return action.lower().replace("mark_", "").strip()


def reset_env():
    global total_reward
    total_reward = 0.0
    state = env.reset()
    return {"observation": state, "reward": 0.0, "total_reward": total_reward, "done": False}


def step_env(action: str):
    global total_reward
    action = normalize_action(action)
    result = env.step(action)
    total_reward += result["reward"]
    return {
        "observation": result["observation"],
        "reward": result["reward"],
        "total_reward": total_reward,
        "done": result["done"],
    }


# -------------------------
# Standard endpoints
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
    return {"observation": env.current_email, "total_reward": total_reward}


# -------------------------
# /tasks  — enumerate all 3 tasks (validator reads this)
# -------------------------
@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "email_classification",
                "description": "Classify an email as spam, important, or social",
                "difficulty": "easy",
                "actions": ["spam", "important", "social"],
            },
            {
                "task_id": "spam_detection",
                "description": "Detect whether an email is spam or not_spam",
                "difficulty": "medium",
                "actions": ["spam", "not_spam"],
            },
            {
                "task_id": "email_priority",
                "description": "Assign priority (urgent, normal, low) to an email",
                "difficulty": "hard",
                "actions": ["urgent", "normal", "low"],
            },
        ]
    }


# -------------------------
# /grader  — run grader for a task (validator calls this)
# Returns scores in 0.0–1.0 for easy/medium/hard difficulties
# -------------------------
@app.post("/grader")
def run_grader(req: GraderRequest):
    task_id = req.task_id.strip().lower()

    if task_id == "email_classification":
        scores = {
            "easy":   grader_module.grade_easy(),
            "medium": grader_module.grade_medium(),
            "hard":   grader_module.grade_hard(),
        }
    elif task_id == "spam_detection":
        scores = {
            "easy":   grader_module.grade_spam_easy(),
            "medium": grader_module.grade_spam_medium(),
            "hard":   grader_module.grade_spam_hard(),
        }
    elif task_id == "email_priority":
        scores = {
            "easy":   grader_module.grade_priority_easy(),
            "medium": grader_module.grade_priority_medium(),
            "hard":   grader_module.grade_priority_hard(),
        }
    else:
        return {"error": f"Unknown task_id: {task_id}"}

    return {
        "task_id": task_id,
        "scores": scores,
        "score": scores["easy"],   # top-level score = easy difficulty
    }


# -------------------------
# /grader GET fallback — runs all tasks at once
# -------------------------
@app.get("/grader")
def run_all_graders():
    return {
        "tasks": [
            {
                "task_id": "email_classification",
                "scores": {
                    "easy":   grader_module.grade_easy(),
                    "medium": grader_module.grade_medium(),
                    "hard":   grader_module.grade_hard(),
                },
            },
            {
                "task_id": "spam_detection",
                "scores": {
                    "easy":   grader_module.grade_spam_easy(),
                    "medium": grader_module.grade_spam_medium(),
                    "hard":   grader_module.grade_spam_hard(),
                },
            },
            {
                "task_id": "email_priority",
                "scores": {
                    "easy":   grader_module.grade_priority_easy(),
                    "medium": grader_module.grade_priority_medium(),
                    "hard":   grader_module.grade_priority_hard(),
                },
            },
        ]
    }


# -------------------------
# Main entry
# -------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
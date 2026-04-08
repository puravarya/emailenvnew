from fastapi import FastAPI
from pydantic import BaseModel
from env import EmailEnv
import grader as grader_module

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
# GRADER ENDPOINT (required by validator)
# Returns scores for all 3 tasks
# -------------------------
@app.get("/grader")
def grader():
    return {
        "tasks": [
            {
                "id": "email_classification",
                "description": "Classify emails as spam, important, or social",
                "scores": {
                    "easy": grader_module.grade_easy(),
                    "medium": grader_module.grade_medium(),
                    "hard": grader_module.grade_hard()
                }
            },
            {
                "id": "spam_detection",
                "description": "Detect whether an email is spam or not spam",
                "scores": {
                    "easy": grader_module.grade_spam_easy(),
                    "medium": grader_module.grade_spam_medium(),
                    "hard": grader_module.grade_spam_hard()
                }
            },
            {
                "id": "email_priority",
                "description": "Assign priority level (urgent, normal, low) to emails",
                "scores": {
                    "easy": grader_module.grade_priority_easy(),
                    "medium": grader_module.grade_priority_medium(),
                    "hard": grader_module.grade_priority_hard()
                }
            }
        ]
    }


# -------------------------
# MAIN ENTRY (for OpenEnv)
# -------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
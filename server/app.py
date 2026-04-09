"""
Email Triage RL Environment - server/app.py
All logic is self-contained here to avoid import path issues in Docker.
"""
import sys
import os

# Add root to path so env.py and grader.py can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(docs_url="/docs", redoc_url=None)


# ============================================================
# TASK DEFINITIONS - all 3 tasks with their reward tables
# ============================================================

TASK_EMAIL_CLASSIFICATION = {
    "task_id": "email_classification",
    "description": "Classify each email as spam, important, or social",
    "difficulty": "easy",
    "actions": ["spam", "important", "social"],
    "test_cases": [
        ("win a lottery now!!!", "spam"),
        ("meeting with ceo tomorrow", "important"),
        ("huge discount just for you", "spam"),
        ("project deadline tomorrow", "important"),
        ("claim your prize now!!!", "spam"),
        ("we have christmas celebration tomorrow at office", "social"),
        ("vogue magazine 2026", "social"),
        ("i-max theatre experience", "social"),
    ],
    "reward_table": {
        "win a lottery now!!!":                              {"spam": 1.0, "social": 0.5, "important": 0.0},
        "meeting with ceo tomorrow":                         {"spam": 0.0, "social": 0.5, "important": 1.0},
        "huge discount just for you":                        {"spam": 1.0, "social": 0.5, "important": 0.0},
        "project deadline tomorrow":                         {"spam": 0.0, "social": 0.5, "important": 1.0},
        "claim your prize now!!!":                           {"spam": 1.0, "social": 0.5, "important": 0.0},
        "we have christmas celebration tomorrow at office":  {"spam": 0.0, "social": 1.0, "important": 0.5},
        "vogue magazine 2026":                               {"spam": 0.5, "social": 1.0, "important": 0.0},
        "i-max theatre experience":                          {"spam": 0.5, "social": 1.0, "important": 0.0},
    },
}

TASK_SPAM_DETECTION = {
    "task_id": "spam_detection",
    "description": "Detect whether an email is spam or not_spam (binary classification)",
    "difficulty": "medium",
    "actions": ["spam", "not_spam"],
    "test_cases": [
        ("click here to win iphone", "spam"),
        ("your invoice is attached", "not_spam"),
        ("congratulations you won $1000", "spam"),
        ("team standup at 10am", "not_spam"),
        ("limited offer buy now", "spam"),
        ("please review the attached report", "not_spam"),
        ("you have been selected for a prize", "spam"),
        ("quarterly review meeting invite", "not_spam"),
    ],
    "reward_table": {
        "click here to win iphone":           {"spam": 1.0, "not_spam": 0.0},
        "your invoice is attached":            {"spam": 0.0, "not_spam": 1.0},
        "congratulations you won $1000":       {"spam": 1.0, "not_spam": 0.0},
        "team standup at 10am":                {"spam": 0.0, "not_spam": 1.0},
        "limited offer buy now":               {"spam": 1.0, "not_spam": 0.0},
        "please review the attached report":   {"spam": 0.0, "not_spam": 1.0},
        "you have been selected for a prize":  {"spam": 1.0, "not_spam": 0.0},
        "quarterly review meeting invite":     {"spam": 0.0, "not_spam": 1.0},
    },
}

TASK_EMAIL_PRIORITY = {
    "task_id": "email_priority",
    "description": "Assign priority level urgent, normal, or low to each email",
    "difficulty": "hard",
    "actions": ["urgent", "normal", "low"],
    "test_cases": [
        ("server is down production issue", "urgent"),
        ("happy birthday wishes", "low"),
        ("client contract needs signature today", "urgent"),
        ("newsletter subscription confirmed", "low"),
        ("critical bug in live system", "urgent"),
        ("weekly team lunch reminder", "low"),
        ("urgent approval needed for budget", "urgent"),
        ("monthly analytics report", "normal"),
    ],
    "reward_table": {
        "server is down production issue":       {"urgent": 1.0, "normal": 0.3, "low": 0.0},
        "happy birthday wishes":                 {"urgent": 0.0, "normal": 0.3, "low": 1.0},
        "client contract needs signature today": {"urgent": 1.0, "normal": 0.3, "low": 0.0},
        "newsletter subscription confirmed":     {"urgent": 0.0, "normal": 0.3, "low": 1.0},
        "critical bug in live system":           {"urgent": 1.0, "normal": 0.3, "low": 0.0},
        "weekly team lunch reminder":            {"urgent": 0.0, "normal": 0.5, "low": 1.0},
        "urgent approval needed for budget":     {"urgent": 1.0, "normal": 0.3, "low": 0.0},
        "monthly analytics report":              {"urgent": 0.0, "normal": 1.0, "low": 0.3},
    },
}

ALL_TASKS = [TASK_EMAIL_CLASSIFICATION, TASK_SPAM_DETECTION, TASK_EMAIL_PRIORITY]
TASK_MAP = {t["task_id"]: t for t in ALL_TASKS}


# ============================================================
# GRADER - deterministic, no randomness
# ============================================================

def _clamp(score: float) -> float:
    """Keep score strictly in (0.0, 1.0)."""
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 4)


def _run_task_grader(task: dict, difficulty: str) -> float:
    """
    Run all test cases for a task and return a score in (0.0, 1.0).
    difficulty controls which scoring mode:
      easy   - average reward across test cases
      medium - fraction of test cases with reward >= 0.5
      hard   - fraction of test cases with reward == 1.0 (perfect only)
    """
    cases = task["test_cases"]
    rewards = task["reward_table"]

    total = 0.0
    perfect = 0
    good = 0

    for text, correct_action in cases:
        r = rewards.get(text, {}).get(correct_action, 0.0)
        total += r
        if r == 1.0:
            perfect += 1
        if r >= 0.5:
            good += 1

    n = len(cases)
    if difficulty == "easy":
        score = total / n
    elif difficulty == "medium":
        score = good / n
    else:  # hard
        score = perfect / n

    return _clamp(score)


def grade_task(task_id: str) -> dict:
    """Return easy/medium/hard scores for a task."""
    task = TASK_MAP.get(task_id)
    if not task:
        return {"error": f"Unknown task_id: {task_id}"}
    return {
        "task_id": task_id,
        "easy":   _run_task_grader(task, "easy"),
        "medium": _run_task_grader(task, "medium"),
        "hard":   _run_task_grader(task, "hard"),
    }


# ============================================================
# ENVIRONMENT (step/reset for the primary task)
# ============================================================

import random

_current_email = None
_current_task_id = "email_classification"
_total_reward = 0.0


class StepRequest(BaseModel):
    action: str
    task_id: str = "email_classification"


class GraderRequest(BaseModel):
    task_id: str


# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return all 3 tasks — required by the hackathon validator."""
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "actions": t["actions"],
            }
            for t in ALL_TASKS
        ]
    }


@app.post("/reset")
def reset(task_id: str = "email_classification"):
    global _current_email, _current_task_id, _total_reward
    _total_reward = 0.0
    _current_task_id = task_id
    task = TASK_MAP.get(task_id, TASK_EMAIL_CLASSIFICATION)
    _current_email = random.choice([tc[0] for tc in task["test_cases"]])
    return {
        "observation": _current_email,
        "task_id": _current_task_id,
        "reward": 0.0,
        "total_reward": 0.0,
        "done": False,
    }


@app.post("/step")
def step(req: StepRequest):
    global _total_reward
    action = req.action.lower().replace("mark_", "").strip()
    task = TASK_MAP.get(req.task_id, TASK_EMAIL_CLASSIFICATION)
    reward = task["reward_table"].get(_current_email or "", {}).get(action, 0.0)
    _total_reward += reward
    return {
        "observation": _current_email,
        "reward": reward,
        "total_reward": _total_reward,
        "done": True,
    }


@app.get("/state")
def state():
    return {
        "observation": _current_email,
        "task_id": _current_task_id,
        "total_reward": _total_reward,
    }


# ============================================================
# GRADER ENDPOINTS - validator calls these
# ============================================================

@app.get("/grader")
def grader_all():
    """GET /grader — run all 3 task graders, return all scores."""
    return {
        "tasks": [
            grade_task(t["task_id"])
            for t in ALL_TASKS
        ]
    }


@app.post("/grader")
def grader_post(req: GraderRequest):
    """POST /grader — run grader for a specific task_id."""
    result = grade_task(req.task_id)
    if "error" in result:
        return result
    return {
        "task_id": result["task_id"],
        "score": result["easy"],   # primary score
        "scores": {
            "easy":   result["easy"],
            "medium": result["medium"],
            "hard":   result["hard"],
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
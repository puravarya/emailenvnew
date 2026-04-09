"""
Email Triage RL Environment — server/app.py
Completely self-contained: all task data, reward logic, and graders are
defined here so there are zero import failures inside Docker.
"""
import random
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(docs_url="/docs", redoc_url=None)

# ──────────────────────────────────────────────────────────────────────────────
# REWARD TABLES (one per task)
# ──────────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_REWARDS = {
    "win a lottery now!!!":                             {"spam": 1.0, "social": 0.5, "important": 0.0},
    "meeting with ceo tomorrow":                        {"spam": 0.0, "social": 0.5, "important": 1.0},
    "huge discount just for you":                       {"spam": 1.0, "social": 0.5, "important": 0.0},
    "project deadline tomorrow":                        {"spam": 0.0, "social": 0.5, "important": 1.0},
    "claim your prize now!!!":                          {"spam": 1.0, "social": 0.5, "important": 0.0},
    "we have christmas celebration tomorrow at office": {"spam": 0.0, "social": 1.0, "important": 0.5},
    "vogue magazine 2026":                              {"spam": 0.5, "social": 1.0, "important": 0.0},
    "i-max theatre experience":                         {"spam": 0.5, "social": 1.0, "important": 0.0},
}

_SPAM_REWARDS = {
    "click here to win iphone":           {"spam": 1.0, "not_spam": 0.0},
    "your invoice is attached":           {"spam": 0.0, "not_spam": 1.0},
    "congratulations you won $1000":      {"spam": 1.0, "not_spam": 0.0},
    "team standup at 10am":               {"spam": 0.0, "not_spam": 1.0},
    "limited offer buy now":              {"spam": 1.0, "not_spam": 0.0},
    "please review the attached report":  {"spam": 0.0, "not_spam": 1.0},
    "you have been selected for a prize": {"spam": 1.0, "not_spam": 0.0},
    "quarterly review meeting invite":    {"spam": 0.0, "not_spam": 1.0},
}

_PRIORITY_REWARDS = {
    "server is down production issue":       {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "happy birthday wishes":                 {"urgent": 0.0, "normal": 0.3, "low": 1.0},
    "client contract needs signature today": {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "newsletter subscription confirmed":     {"urgent": 0.0, "normal": 0.3, "low": 1.0},
    "critical bug in live system":           {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "weekly team lunch reminder":            {"urgent": 0.0, "normal": 0.5, "low": 1.0},
    "urgent approval needed for budget":     {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "monthly analytics report":              {"urgent": 0.0, "normal": 1.0, "low": 0.3},
}

# Fixed test-case pairs used by graders (deterministic — no randomness)
_CLASSIFICATION_CASES = [
    ("win a lottery now!!!", "spam"),
    ("meeting with ceo tomorrow", "important"),
    ("huge discount just for you", "spam"),
    ("project deadline tomorrow", "important"),
    ("claim your prize now!!!", "spam"),
    ("we have christmas celebration tomorrow at office", "social"),
    ("vogue magazine 2026", "social"),
    ("i-max theatre experience", "social"),
]
_SPAM_CASES = [
    ("click here to win iphone", "spam"),
    ("your invoice is attached", "not_spam"),
    ("congratulations you won $1000", "spam"),
    ("team standup at 10am", "not_spam"),
    ("limited offer buy now", "spam"),
    ("please review the attached report", "not_spam"),
    ("you have been selected for a prize", "spam"),
    ("quarterly review meeting invite", "not_spam"),
]
_PRIORITY_CASES = [
    ("server is down production issue", "urgent"),
    ("happy birthday wishes", "low"),
    ("client contract needs signature today", "urgent"),
    ("newsletter subscription confirmed", "low"),
    ("critical bug in live system", "urgent"),
    ("weekly team lunch reminder", "low"),
    ("urgent approval needed for budget", "urgent"),
    ("monthly analytics report", "normal"),
]

# ──────────────────────────────────────────────────────────────────────────────
# GRADER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(score: float) -> float:
    """Keep score strictly inside (0.0, 1.0) as required by the spec."""
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 4)


def _grade(cases, reward_table, mode: str) -> float:
    """
    Run fixed test cases against the oracle (correct) action.
    mode='easy'   → average reward (partial credit allowed)
    mode='medium' → fraction with reward >= 0.5
    mode='hard'   → fraction with reward == 1.0 (perfect only)
    """
    total, perfect, good = 0.0, 0, 0
    for text, action in cases:
        r = reward_table.get(text, {}).get(action, 0.0)
        total += r
        if r == 1.0:
            perfect += 1
        if r >= 0.5:
            good += 1
    n = len(cases)
    if mode == "easy":
        return _clamp(total / n)
    if mode == "medium":
        return _clamp(good / n)
    return _clamp(perfect / n)   # hard


def _grade_all(cases, reward_table) -> dict:
    return {
        "easy":   _grade(cases, reward_table, "easy"),
        "medium": _grade(cases, reward_table, "medium"),
        "hard":   _grade(cases, reward_table, "hard"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────

_state = {
    "current_email": None,
    "task_id": "email_classification",
    "total_reward": 0.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ──────────────────────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action: str
    task_id: str = "email_classification"

class GraderRequest(BaseModel):
    task_id: str

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str = "email_classification"):
    _state["task_id"] = task_id
    _state["total_reward"] = 0.0
    pool = {
        "email_classification": list(_CLASSIFICATION_REWARDS.keys()),
        "spam_detection":       list(_SPAM_REWARDS.keys()),
        "email_priority":       list(_PRIORITY_REWARDS.keys()),
    }.get(task_id, list(_CLASSIFICATION_REWARDS.keys()))
    _state["current_email"] = random.choice(pool)
    return {
        "observation": _state["current_email"],
        "task_id": task_id,
        "reward": 0.0,
        "total_reward": 0.0,
        "done": False,
    }


@app.post("/step")
def step(req: StepRequest):
    action = req.action.lower().replace("mark_", "").strip()
    table = {
        "email_classification": _CLASSIFICATION_REWARDS,
        "spam_detection":       _SPAM_REWARDS,
        "email_priority":       _PRIORITY_REWARDS,
    }.get(req.task_id, _CLASSIFICATION_REWARDS)
    reward = table.get(_state["current_email"] or "", {}).get(action, 0.0)
    _state["total_reward"] += reward
    return {
        "observation": _state["current_email"],
        "reward": reward,
        "total_reward": _state["total_reward"],
        "done": True,
    }


@app.get("/state")
def state():
    return {
        "observation": _state["current_email"],
        "task_id": _state["task_id"],
        "total_reward": _state["total_reward"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# /tasks  ← validator enumerates tasks here
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "email_classification",
                "description": "Classify each email as spam, important, or social",
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
                "description": "Assign priority level urgent, normal, or low to an email",
                "difficulty": "hard",
                "actions": ["urgent", "normal", "low"],
            },
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# /grader  ← validator runs graders here (GET + POST both supported)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_grader(task_id: str):
    if task_id == "email_classification":
        scores = _grade_all(_CLASSIFICATION_CASES, _CLASSIFICATION_REWARDS)
    elif task_id == "spam_detection":
        scores = _grade_all(_SPAM_CASES, _SPAM_REWARDS)
    elif task_id == "email_priority":
        scores = _grade_all(_PRIORITY_CASES, _PRIORITY_REWARDS)
    else:
        return None
    return {
        "task_id": task_id,
        "score": scores["easy"],
        "scores": scores,
    }


@app.get("/grader")
def grader_get():
    """Run all 3 graders — validator may call this."""
    results = []
    for tid in ["email_classification", "spam_detection", "email_priority"]:
        results.append(_compute_grader(tid))
    return {"tasks": results}


@app.post("/grader")
def grader_post(req: GraderRequest):
    """Run grader for a specific task — validator may call this."""
    result = _compute_grader(req.task_id)
    if result is None:
        return {"error": f"Unknown task_id: {req.task_id}"}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
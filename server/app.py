"""
Email Triage RL Environment — server/app.py
Fully self-contained: no imports of env.py or grader.py.
All reward tables, grader logic, and task definitions are inline.
"""
import random
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(docs_url="/docs", redoc_url=None)

# ── Reward tables ──────────────────────────────────────────────────────────────

_R_CLASS = {
    "win a lottery now!!!":                             {"spam": 1.0, "social": 0.5, "important": 0.0},
    "meeting with ceo tomorrow":                        {"spam": 0.0, "social": 0.5, "important": 1.0},
    "huge discount just for you":                       {"spam": 1.0, "social": 0.5, "important": 0.0},
    "project deadline tomorrow":                        {"spam": 0.0, "social": 0.5, "important": 1.0},
    "claim your prize now!!!":                          {"spam": 1.0, "social": 0.5, "important": 0.0},
    "we have christmas celebration tomorrow at office": {"spam": 0.0, "social": 1.0, "important": 0.5},
    "vogue magazine 2026":                              {"spam": 0.5, "social": 1.0, "important": 0.0},
    "i-max theatre experience":                         {"spam": 0.5, "social": 1.0, "important": 0.0},
}
_R_SPAM = {
    "click here to win iphone":           {"spam": 1.0, "not_spam": 0.0},
    "your invoice is attached":           {"spam": 0.0, "not_spam": 1.0},
    "congratulations you won $1000":      {"spam": 1.0, "not_spam": 0.0},
    "team standup at 10am":               {"spam": 0.0, "not_spam": 1.0},
    "limited offer buy now":              {"spam": 1.0, "not_spam": 0.0},
    "please review the attached report":  {"spam": 0.0, "not_spam": 1.0},
    "you have been selected for a prize": {"spam": 1.0, "not_spam": 0.0},
    "quarterly review meeting invite":    {"spam": 0.0, "not_spam": 1.0},
}
_R_PRIO = {
    "server is down production issue":       {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "happy birthday wishes":                 {"urgent": 0.0, "normal": 0.3, "low": 1.0},
    "client contract needs signature today": {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "newsletter subscription confirmed":     {"urgent": 0.0, "normal": 0.3, "low": 1.0},
    "critical bug in live system":           {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "weekly team lunch reminder":            {"urgent": 0.0, "normal": 0.5, "low": 1.0},
    "urgent approval needed for budget":     {"urgent": 1.0, "normal": 0.3, "low": 0.0},
    "monthly analytics report":              {"urgent": 0.0, "normal": 1.0, "low": 0.3},
}

# ── Fixed test cases (deterministic — no randomness) ──────────────────────────

_C_CLASS = [
    ("win a lottery now!!!", "spam"),
    ("meeting with ceo tomorrow", "important"),
    ("huge discount just for you", "spam"),
    ("project deadline tomorrow", "important"),
    ("claim your prize now!!!", "spam"),
    ("we have christmas celebration tomorrow at office", "social"),
    ("vogue magazine 2026", "social"),
    ("i-max theatre experience", "social"),
]
_C_SPAM = [
    ("click here to win iphone", "spam"),
    ("your invoice is attached", "not_spam"),
    ("congratulations you won $1000", "spam"),
    ("team standup at 10am", "not_spam"),
    ("limited offer buy now", "spam"),
    ("please review the attached report", "not_spam"),
    ("you have been selected for a prize", "spam"),
    ("quarterly review meeting invite", "not_spam"),
]
_C_PRIO = [
    ("server is down production issue", "urgent"),
    ("happy birthday wishes", "low"),
    ("client contract needs signature today", "urgent"),
    ("newsletter subscription confirmed", "low"),
    ("critical bug in live system", "urgent"),
    ("weekly team lunch reminder", "low"),
    ("urgent approval needed for budget", "urgent"),
    ("monthly analytics report", "normal"),
]

# ── Grader ────────────────────────────────────────────────────────────────────

def _clamp(s: float) -> float:
    if s <= 0.0: return 0.01
    if s >= 1.0: return 0.99
    return round(s, 4)

def _score(cases, table, mode: str) -> float:
    total, perfect, good = 0.0, 0, 0
    for text, action in cases:
        r = table.get(text, {}).get(action, 0.0)
        total += r
        if r == 1.0: perfect += 1
        if r >= 0.5: good += 1
    n = len(cases)
    if mode == "easy":   return _clamp(total / n)
    if mode == "medium": return _clamp(good / n)
    return _clamp(perfect / n)

def _grades(cases, table):
    return {
        "easy":   _score(cases, table, "easy"),
        "medium": _score(cases, table, "medium"),
        "hard":   _score(cases, table, "hard"),
    }

_GRADER_MAP = {
    "email_classification": (_C_CLASS, _R_CLASS),
    "spam_detection":       (_C_SPAM,  _R_SPAM),
    "email_priority":       (_C_PRIO,  _R_PRIO),
}

# ── State ─────────────────────────────────────────────────────────────────────

_s = {"email": None, "task_id": "email_classification", "total": 0.0}

# ── Request models ────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action: str
    task_id: str = "email_classification"

class GraderRequest(BaseModel):
    task_id: str

# ── Standard endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_id: str = "email_classification"):
    _s["task_id"] = task_id
    _s["total"] = 0.0
    table = _GRADER_MAP.get(task_id, (_C_CLASS, _R_CLASS))[1]
    _s["email"] = random.choice(list(table.keys()))
    return {"observation": _s["email"], "task_id": task_id,
            "reward": 0.0, "total_reward": 0.0, "done": False}

@app.post("/step")
def step(req: StepRequest):
    action = req.action.lower().replace("mark_", "").strip()
    table = _GRADER_MAP.get(req.task_id, (_C_CLASS, _R_CLASS))[1]
    reward = table.get(_s["email"] or "", {}).get(action, 0.0)
    _s["total"] += reward
    return {"observation": _s["email"], "reward": reward,
            "total_reward": _s["total"], "done": True}

@app.get("/state")
def state():
    return {"observation": _s["email"], "task_id": _s["task_id"],
            "total_reward": _s["total"]}

# ── /tasks ────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {"tasks": [
        {"task_id": "email_classification",
         "description": "Classify each email as spam, important, or social",
         "difficulty": "easy",
         "actions": ["spam", "important", "social"]},
        {"task_id": "spam_detection",
         "description": "Detect whether an email is spam or not_spam",
         "difficulty": "medium",
         "actions": ["spam", "not_spam"]},
        {"task_id": "email_priority",
         "description": "Assign priority level urgent, normal, or low to an email",
         "difficulty": "hard",
         "actions": ["urgent", "normal", "low"]},
    ]}

# ── /grader ───────────────────────────────────────────────────────────────────

def _run_grader(task_id: str):
    pair = _GRADER_MAP.get(task_id)
    if pair is None:
        return {"error": f"Unknown task_id: {task_id}"}
    cases, table = pair
    scores = _grades(cases, table)
    return {"task_id": task_id, "score": scores["easy"], "scores": scores}

@app.get("/grader")
def grader_get():
    return {"tasks": [_run_grader(tid) for tid in _GRADER_MAP]}

@app.post("/grader")
def grader_post(req: GraderRequest):
    return _run_grader(req.task_id)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
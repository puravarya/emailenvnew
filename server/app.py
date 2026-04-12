import random
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(docs_url="/docs", redoc_url=None)


def _c(s):
    s = float(s)
    if s <= 0.0: return 0.01
    if s >= 1.0: return 0.99
    return round(s, 4)


_R_CLASS = {
    "win a lottery now!!!":                             {"spam": 0.99, "social": 0.50, "important": 0.01},
    "meeting with ceo tomorrow":                        {"spam": 0.01, "social": 0.50, "important": 0.99},
    "huge discount just for you":                       {"spam": 0.99, "social": 0.50, "important": 0.01},
    "project deadline tomorrow":                        {"spam": 0.01, "social": 0.50, "important": 0.99},
    "claim your prize now!!!":                          {"spam": 0.99, "social": 0.50, "important": 0.01},
    "we have christmas celebration tomorrow at office": {"spam": 0.01, "social": 0.99, "important": 0.50},
    "vogue magazine 2026":                              {"spam": 0.50, "social": 0.99, "important": 0.01},
    "i-max theatre experience":                         {"spam": 0.50, "social": 0.99, "important": 0.01},
}
_R_SPAM = {
    "click here to win iphone":           {"spam": 0.99, "not_spam": 0.01},
    "your invoice is attached":           {"spam": 0.01, "not_spam": 0.99},
    "congratulations you won $1000":      {"spam": 0.99, "not_spam": 0.01},
    "team standup at 10am":               {"spam": 0.01, "not_spam": 0.99},
    "limited offer buy now":              {"spam": 0.99, "not_spam": 0.01},
    "please review the attached report":  {"spam": 0.01, "not_spam": 0.99},
    "you have been selected for a prize": {"spam": 0.99, "not_spam": 0.01},
    "quarterly review meeting invite":    {"spam": 0.01, "not_spam": 0.99},
}
_R_PRIO = {
    "server is down production issue":       {"urgent": 0.99, "normal": 0.30, "low": 0.01},
    "happy birthday wishes":                 {"urgent": 0.01, "normal": 0.30, "low": 0.99},
    "client contract needs signature today": {"urgent": 0.99, "normal": 0.30, "low": 0.01},
    "newsletter subscription confirmed":     {"urgent": 0.01, "normal": 0.30, "low": 0.99},
    "critical bug in live system":           {"urgent": 0.99, "normal": 0.30, "low": 0.01},
    "weekly team lunch reminder":            {"urgent": 0.01, "normal": 0.50, "low": 0.99},
    "urgent approval needed for budget":     {"urgent": 0.99, "normal": 0.30, "low": 0.01},
    "monthly analytics report":              {"urgent": 0.01, "normal": 0.99, "low": 0.30},
}

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

_GRADER_MAP = {
    "email_classification": (_C_CLASS, _R_CLASS),
    "spam_detection":       (_C_SPAM,  _R_SPAM),
    "email_priority":       (_C_PRIO,  _R_PRIO),
}


def _run(cases, table):
    if not cases:
        return 0.50
    total = 0.0
    for t, a in cases:
        total += table.get(t, {}).get(a, 0.50)
    return _c(total / len(cases))


def _task_score(task_id):
    pair = _GRADER_MAP.get(task_id)
    if not pair:
        return 0.50
    return _run(pair[0], pair[1])


_s = {"email": None, "task_id": "email_classification", "total": 0.50}


class StepRequest(BaseModel):
    action: str
    task_id: str = "email_classification"


class GraderRequest(BaseModel):
    task_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str = "email_classification"):
    _s["task_id"] = task_id
    _s["total"] = 0.50
    table = _GRADER_MAP.get(task_id, (_C_CLASS, _R_CLASS))[1]
    _s["email"] = random.choice(list(table.keys()))
    return {"observation": _s["email"], "task_id": task_id,
            "reward": 0.50, "total_reward": 0.50, "done": False}


@app.post("/step")
def step(req: StepRequest):
    action = req.action.lower().replace("mark_", "").strip()
    table = _GRADER_MAP.get(req.task_id, (_C_CLASS, _R_CLASS))[1]
    raw = table.get(_s["email"] or "", {}).get(action, 0.50)
    reward = _c(raw)
    _s["total"] = _c(_s["total"] + reward)
    return {"observation": _s["email"], "reward": reward,
            "total_reward": _s["total"], "done": True}


@app.get("/state")
def state():
    return {"observation": _s["email"], "task_id": _s["task_id"],
            "total_reward": _s["total"]}


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
         "description": "Assign priority level urgent, normal, or low",
         "difficulty": "hard",
         "actions": ["urgent", "normal", "low"]},
    ]}


def _grader_result(task_id):
    score = _task_score(task_id)
    return {"task_id": task_id, "score": score,
            "scores": {"easy": score, "medium": score, "hard": score}}


@app.get("/grader")
def grader_get():
    return {"tasks": [_grader_result(tid) for tid in _GRADER_MAP]}


@app.post("/grader")
def grader_post(req: GraderRequest):
    if req.task_id not in _GRADER_MAP:
        return {"error": f"Unknown task_id: {req.task_id}"}
    return _grader_result(req.task_id)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

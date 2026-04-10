"""
grader.py — Deterministic graders for all 3 tasks.
All scores strictly in (0.0, 1.0) — never exactly 0.0 or 1.0.
"""


def _clamp(s: float) -> float:
    s = float(s)
    if s <= 0.0: return 0.01
    if s >= 1.0: return 0.99
    return round(s, 4)


_R_CLASS = {
    "win a lottery now!!!":                             {"spam": 0.99, "social": 0.5,  "important": 0.01},
    "meeting with ceo tomorrow":                        {"spam": 0.01, "social": 0.5,  "important": 0.99},
    "huge discount just for you":                       {"spam": 0.99, "social": 0.5,  "important": 0.01},
    "project deadline tomorrow":                        {"spam": 0.01, "social": 0.5,  "important": 0.99},
    "claim your prize now!!!":                          {"spam": 0.99, "social": 0.5,  "important": 0.01},
    "we have christmas celebration tomorrow at office": {"spam": 0.01, "social": 0.99, "important": 0.5},
    "vogue magazine 2026":                              {"spam": 0.5,  "social": 0.99, "important": 0.01},
    "i-max theatre experience":                         {"spam": 0.5,  "social": 0.99, "important": 0.01},
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
    "server is down production issue":       {"urgent": 0.99, "normal": 0.3,  "low": 0.01},
    "happy birthday wishes":                 {"urgent": 0.01, "normal": 0.3,  "low": 0.99},
    "client contract needs signature today": {"urgent": 0.99, "normal": 0.3,  "low": 0.01},
    "newsletter subscription confirmed":     {"urgent": 0.01, "normal": 0.3,  "low": 0.99},
    "critical bug in live system":           {"urgent": 0.99, "normal": 0.3,  "low": 0.01},
    "weekly team lunch reminder":            {"urgent": 0.01, "normal": 0.5,  "low": 0.99},
    "urgent approval needed for budget":     {"urgent": 0.99, "normal": 0.3,  "low": 0.01},
    "monthly analytics report":              {"urgent": 0.01, "normal": 0.99, "low": 0.3},
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


def _run(cases, table) -> float:
    if not cases:
        return 0.01
    total = 0.0
    for t, a in cases:
        total += table.get(t, {}).get(a, 0.01)
    return _clamp(total / len(cases))


def grade_email_classification() -> float:
    return _run(_C_CLASS, _R_CLASS)


def grade_spam_detection() -> float:
    return _run(_C_SPAM, _R_SPAM)


def grade_email_priority() -> float:
    return _run(_C_PRIO, _R_PRIO)


# Legacy aliases (for backward compatibility)
def grade_easy():            return grade_email_classification()
def grade_medium():          return grade_email_classification()
def grade_hard():            return grade_email_classification()
def grade_spam_easy():       return grade_spam_detection()
def grade_spam_medium():     return grade_spam_detection()
def grade_spam_hard():       return grade_spam_detection()
def grade_priority_easy():   return grade_email_priority()
def grade_priority_medium(): return grade_email_priority()
def grade_priority_hard():   return grade_email_priority()


TASKS = [
    {
        "task_id":     "email_classification",
        "description": "Classify each email as spam, important, or social",
        "difficulty":  "easy",
        "actions":     ["spam", "important", "social"],
        "grader_fn":   grade_email_classification,
    },
    {
        "task_id":     "spam_detection",
        "description": "Detect whether an email is spam or not_spam",
        "difficulty":  "medium",
        "actions":     ["spam", "not_spam"],
        "grader_fn":   grade_spam_detection,
    },
    {
        "task_id":     "email_priority",
        "description": "Assign priority level urgent, normal, or low",
        "difficulty":  "hard",
        "actions":     ["urgent", "normal", "low"],
        "grader_fn":   grade_email_priority,
    },
]


if __name__ == "__main__":
    for task in TASKS:
        score = task["grader_fn"]()
        assert 0.0 < score < 1.0, f"FAIL: {task['task_id']} = {score}"
        print(f"PASS  {task['task_id']} = {score}")
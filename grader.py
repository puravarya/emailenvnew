"""
Deterministic graders for all 3 tasks.
Each grader runs fixed test cases and returns a score in (0.0, 1.0).
No randomness - same inputs, same outputs every time.
"""

from env import EmailEnv, SpamDetectorEnv, EmailPriorityEnv


def _clamp(score):
    """Force score strictly inside (0.0, 1.0)."""
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(float(score), 4)


# ==============================================================
# TASK 1: Email Classification  (spam / important / social)
# ==============================================================

# Fixed test cases - same order every time (deterministic)
TASK1_CASES = [
    # (email_text, correct_action)
    ("win a lottery now!!!", "spam"),
    ("meeting with ceo tomorrow", "important"),
    ("huge discount just for you", "spam"),
    ("project deadline tomorrow", "important"),
    ("claim your prize now!!!", "spam"),
    ("we have christmas celebration tomorrow at office", "social"),
    ("vogue magazine 2026", "social"),
    ("i-max theatre experience", "social"),
]


def grade_easy():
    """
    Task 1 Easy: Classify obvious spam vs important emails.
    Only uses clear-cut cases with strong keyword signals.
    """
    env = EmailEnv()
    cases = [
        ("win a lottery now!!!", "spam"),
        ("meeting with ceo tomorrow", "important"),
        ("huge discount just for you", "spam"),
        ("project deadline tomorrow", "important"),
        ("claim your prize now!!!", "spam"),
    ]
    total_reward = 0.0
    for text, action in cases:
        env.reset(text)
        result = env.step(action)
        total_reward += result["reward"]
    score = total_reward / len(cases)
    return _clamp(score)


def grade_medium():
    """
    Task 1 Medium: All 8 emails including social category.
    Requires distinguishing social from spam and important.
    """
    env = EmailEnv()
    total_reward = 0.0
    for text, action in TASK1_CASES:
        env.reset(text)
        result = env.step(action)
        total_reward += result["reward"]
    score = total_reward / len(TASK1_CASES)
    return _clamp(score)


def grade_hard():
    """
    Task 1 Hard: Penalizes wrong classifications heavily.
    Score only counts emails classified correctly (reward == 1.0).
    Partial-credit (social=0.5) does not count as correct here.
    """
    env = EmailEnv()
    perfect = 0
    for text, action in TASK1_CASES:
        env.reset(text)
        result = env.step(action)
        if result["reward"] == 1.0:
            perfect += 1
    score = perfect / len(TASK1_CASES)
    return _clamp(score)


# ==============================================================
# TASK 2: Spam Detection  (spam / not_spam)
# ==============================================================

TASK2_CASES = [
    ("click here to win iphone", "spam"),
    ("your invoice is attached", "not_spam"),
    ("congratulations you won $1000", "spam"),
    ("team standup at 10am", "not_spam"),
    ("limited offer buy now", "spam"),
    ("please review the attached report", "not_spam"),
    ("you have been selected for a prize", "spam"),
    ("quarterly review meeting invite", "not_spam"),
]


def grade_spam_easy():
    """
    Task 2 Easy: Classify 4 obvious spam/not-spam emails.
    """
    env = SpamDetectorEnv()
    cases = TASK2_CASES[:4]
    total_reward = 0.0
    for text, action in cases:
        env.reset(text)
        result = env.step(action)
        total_reward += result["reward"]
    score = total_reward / len(cases)
    return _clamp(score)


def grade_spam_medium():
    """
    Task 2 Medium: All 8 spam/not-spam emails.
    """
    env = SpamDetectorEnv()
    total_reward = 0.0
    for text, action in TASK2_CASES:
        env.reset(text)
        result = env.step(action)
        total_reward += result["reward"]
    score = total_reward / len(TASK2_CASES)
    return _clamp(score)


def grade_spam_hard():
    """
    Task 2 Hard: Requires perfect binary classification.
    Any wrong answer (false positive or false negative) reduces score.
    """
    env = SpamDetectorEnv()
    correct = 0
    for text, action in TASK2_CASES:
        env.reset(text)
        result = env.step(action)
        if result["reward"] == 1.0:
            correct += 1
    score = correct / len(TASK2_CASES)
    return _clamp(score)


# ==============================================================
# TASK 3: Email Priority  (urgent / normal / low)
# ==============================================================

TASK3_CASES = [
    ("server is down production issue", "urgent"),
    ("happy birthday wishes", "low"),
    ("client contract needs signature today", "urgent"),
    ("newsletter subscription confirmed", "low"),
    ("critical bug in live system", "urgent"),
    ("weekly team lunch reminder", "low"),
    ("urgent approval needed for budget", "urgent"),
    ("monthly analytics report", "normal"),
]


def grade_priority_easy():
    """
    Task 3 Easy: 4 cases with very obvious urgency signals.
    """
    env = EmailPriorityEnv()
    cases = [
        ("server is down production issue", "urgent"),
        ("happy birthday wishes", "low"),
        ("critical bug in live system", "urgent"),
        ("newsletter subscription confirmed", "low"),
    ]
    total_reward = 0.0
    for text, action in cases:
        env.reset(text)
        result = env.step(action)
        total_reward += result["reward"]
    score = total_reward / len(cases)
    return _clamp(score)


def grade_priority_medium():
    """
    Task 3 Medium: All 8 priority cases including normal.
    """
    env = EmailPriorityEnv()
    total_reward = 0.0
    for text, action in TASK3_CASES:
        env.reset(text)
        result = env.step(action)
        total_reward += result["reward"]
    score = total_reward / len(TASK3_CASES)
    return _clamp(score)


def grade_priority_hard():
    """
    Task 3 Hard: Must correctly distinguish urgent/normal/low perfectly.
    Only counts reward == 1.0 (exact correct label).
    """
    env = EmailPriorityEnv()
    perfect = 0
    for text, action in TASK3_CASES:
        env.reset(text)
        result = env.step(action)
        if result["reward"] == 1.0:
            perfect += 1
    score = perfect / len(TASK3_CASES)
    return _clamp(score)


# ==============================================================
# Run all graders (for testing)
# ==============================================================
if __name__ == "__main__":
    print("=== TASK 1: Email Classification ===")
    print(f"  Easy:   {grade_easy()}")
    print(f"  Medium: {grade_medium()}")
    print(f"  Hard:   {grade_hard()}")

    print("\n=== TASK 2: Spam Detection ===")
    print(f"  Easy:   {grade_spam_easy()}")
    print(f"  Medium: {grade_spam_medium()}")
    print(f"  Hard:   {grade_spam_hard()}")

    print("\n=== TASK 3: Email Priority ===")
    print(f"  Easy:   {grade_priority_easy()}")
    print(f"  Medium: {grade_priority_medium()}")
    print(f"  Hard:   {grade_priority_hard()}")
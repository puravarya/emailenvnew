from env import EmailEnv


# -------------------------
# SCORE NORMALIZER
# -------------------------
def normalize(score):
    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99
    return float(score)


# -------------------------
# TASK 1 - EASY
# -------------------------
def task_easy():
    env = EmailEnv()
    correct = 0
    total = 5

    for _ in range(total):
        text = env.reset()

        if "lottery" in text or "discount" in text:
            action = "spam"
        else:
            action = "important"

        result = env.step(action)

        if result["reward"] > 0:
            correct += 1

    score = correct / total
    return normalize(score)


# -------------------------
# TASK 2 - MEDIUM
# -------------------------
def task_medium():
    env = EmailEnv()
    total_reward = 0
    steps = 5

    for _ in range(steps):
        text = env.reset()

        if "ceo" in text or "deadline" in text:
            action = "important"
        else:
            action = "spam"

        result = env.step(action)
        total_reward += result["reward"]

    score = (total_reward + steps) / (2 * steps)
    return normalize(score)


# -------------------------
# TASK 3 - HARD
# -------------------------
def task_hard():
    env = EmailEnv()
    mistakes = 0
    steps = 5

    for _ in range(steps):
        text = env.reset()

        if len(text) > 25:
            action = "important"
        else:
            action = "spam"

        result = env.step(action)

        if result["reward"] < 0:
            mistakes += 1

    score = 1 - (mistakes / steps)
    return normalize(score)


# -------------------------
# REQUIRED EXPORT (IMPORTANT)
# -------------------------
TASKS = {
    "easy": task_easy,
    "medium": task_medium,
    "hard": task_hard,
}
from env import EmailEnv


# -------------------------
# SAFE SCORE FUNCTION
# -------------------------
def normalize_score(score):
    # force into (0,1) range
    if score <= 0:
        return 0.1
    elif score >= 1:
        return 0.9
    return score


# -------------------------
# EASY
# -------------------------
def grade_easy():
    env = EmailEnv()
    text = env.reset()

    correct = 0
    total = 5

    for _ in range(total):
        if "lottery" in text or "discount" in text:
            action = "spam"
        else:
            action = "important"

        result = env.step(action)

        if result["reward"] > 0:
            correct += 1

    score = correct / total
    return normalize_score(score)


# -------------------------
# MEDIUM
# -------------------------
def grade_medium():
    env = EmailEnv()
    text = env.reset()

    total_reward = 0
    steps = 5

    for _ in range(steps):
        if "ceo" in text or "deadline" in text:
            action = "important"
        else:
            action = "spam"

        result = env.step(action)
        total_reward += result["reward"]

    score = (total_reward + steps) / (2 * steps)  # normalize roughly
    return normalize_score(score)


# -------------------------
# HARD
# -------------------------
def grade_hard():
    env = EmailEnv()
    text = env.reset()

    mistakes = 0
    steps = 5

    for _ in range(steps):
        if len(text) > 25:
            action = "important"
        else:
            action = "spam"

        result = env.step(action)

        if result["reward"] < 0:
            mistakes += 1

    score = 1 - (mistakes / steps)
    return normalize_score(score)
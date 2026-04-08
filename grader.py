from env import EmailEnv


def clamp(score):
    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99
    return float(score)


def grade_easy():
    env = EmailEnv()
    total = 5
    correct = 0

    for _ in range(total):
        text = env.reset()

        if "lottery" in text or "discount" in text:
            action = "spam"
        else:
            action = "important"

        result = env.step(action)

        if result["reward"] > 0:
            correct += 1

    return clamp(correct / total)


def grade_medium():
    env = EmailEnv()
    steps = 5
    total_reward = 0

    for _ in range(steps):
        text = env.reset()

        if "ceo" in text or "deadline" in text:
            action = "important"
        else:
            action = "spam"

        result = env.step(action)
        total_reward += result["reward"]

    score = (total_reward + steps) / (2 * steps)
    return clamp(score)


def grade_hard():
    env = EmailEnv()
    steps = 5
    mistakes = 0

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
    return clamp(score)
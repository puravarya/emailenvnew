from env import EmailEnv, SpamDetectorEnv, EmailPriorityEnv


# -------------------------
# FORCE SCORE IN (0,1)
# -------------------------
def clamp(score):
    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99
    return float(score)


# ==============================================
# TASK 1: Email Classification (spam/important/social)
# ==============================================

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

    score = correct / total
    return clamp(score)


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


# ==============================================
# TASK 2: Spam Detection (spam/not_spam)
# ==============================================

def grade_spam_easy():
    env = SpamDetectorEnv()
    total = 5
    correct = 0

    for _ in range(total):
        text = env.reset()

        if "win" in text or "prize" in text or "offer" in text or "congratulations" in text:
            action = "spam"
        else:
            action = "not_spam"

        result = env.step(action)

        if result["reward"] > 0:
            correct += 1

    score = correct / total
    return clamp(score)


def grade_spam_medium():
    env = SpamDetectorEnv()
    steps = 5
    total_reward = 0

    for _ in range(steps):
        text = env.reset()

        spam_keywords = ["win", "prize", "click", "offer", "selected", "congratulations"]
        if any(kw in text for kw in spam_keywords):
            action = "spam"
        else:
            action = "not_spam"

        result = env.step(action)
        total_reward += result["reward"]

    score = (total_reward + steps) / (2 * steps)
    return clamp(score)


def grade_spam_hard():
    env = SpamDetectorEnv()
    steps = 5
    mistakes = 0

    for _ in range(steps):
        text = env.reset()

        # Harder heuristic: only flag if multiple spam signals
        spam_keywords = ["win", "prize", "click", "offer", "selected", "congratulations", "limited", "iphone", "$"]
        spam_count = sum(1 for kw in spam_keywords if kw in text)

        if spam_count >= 1:
            action = "spam"
        else:
            action = "not_spam"

        result = env.step(action)

        if result["reward"] < 0:
            mistakes += 1

    score = 1 - (mistakes / steps)
    return clamp(score)


# ==============================================
# TASK 3: Email Priority (urgent/normal/low)
# ==============================================

def grade_priority_easy():
    env = EmailPriorityEnv()
    total = 5
    correct = 0

    for _ in range(total):
        text = env.reset()

        if "urgent" in text or "critical" in text or "down" in text:
            action = "urgent"
        else:
            action = "low"

        result = env.step(action)

        if result["reward"] > 0:
            correct += 1

    score = correct / total
    return clamp(score)


def grade_priority_medium():
    env = EmailPriorityEnv()
    steps = 5
    total_reward = 0

    for _ in range(steps):
        text = env.reset()

        urgent_keywords = ["urgent", "critical", "down", "signature", "bug", "approval"]
        low_keywords = ["birthday", "newsletter", "lunch", "wishes"]

        if any(kw in text for kw in urgent_keywords):
            action = "urgent"
        elif any(kw in text for kw in low_keywords):
            action = "low"
        else:
            action = "normal"

        result = env.step(action)
        total_reward += result["reward"]

    score = (total_reward + steps) / (2 * steps)
    return clamp(score)


def grade_priority_hard():
    env = EmailPriorityEnv()
    steps = 5
    mistakes = 0

    for _ in range(steps):
        text = env.reset()

        urgent_keywords = ["urgent", "critical", "down", "signature today", "bug", "approval"]
        low_keywords = ["birthday", "newsletter", "lunch", "wishes", "confirmed"]

        if any(kw in text for kw in urgent_keywords):
            action = "urgent"
        elif any(kw in text for kw in low_keywords):
            action = "low"
        else:
            action = "normal"

        result = env.step(action)

        if result["reward"] < 0:
            mistakes += 1

    score = 1 - (mistakes / steps)
    return clamp(score)


# ==============================================
# RUN ALL GRADERS
# ==============================================
if __name__ == "__main__":
    print("=== TASK 1: Email Classification ===")
    print(f"Easy:   {grade_easy()}")
    print(f"Medium: {grade_medium()}")
    print(f"Hard:   {grade_hard()}")

    print("\n=== TASK 2: Spam Detection ===")
    print(f"Easy:   {grade_spam_easy()}")
    print(f"Medium: {grade_spam_medium()}")
    print(f"Hard:   {grade_spam_hard()}")

    print("\n=== TASK 3: Email Priority ===")
    print(f"Easy:   {grade_priority_easy()}")
    print(f"Medium: {grade_priority_medium()}")
    print(f"Hard:   {grade_priority_hard()}")
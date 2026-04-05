from env import EmailEnv, Action

def grade_easy():
    env = EmailEnv()
    obs = env.reset()

    correct = 0
    total = 0

    while True:
        text = obs.text.lower()

        if "lottery" in text or "discount" in text or "free" in text:
            action = "mark_spam"
        else:
            action = "mark_important"

        obs, reward, done, _ = env.step(Action(action=action))

        if reward.reward > 0:
            correct += 1

        total += 1

        if done:
            break

    return correct / total


def grade_medium():
    env = EmailEnv()
    obs = env.reset()

    total_reward = 0

    while True:
        text = obs.text.lower()

        if "ceo" in text or "deadline" in text:
            action = "mark_important"
        else:
            action = "mark_spam"

        obs, reward, done, _ = env.step(Action(action=action))
        total_reward += reward.reward

        if done:
            break

    return total_reward / 5


def grade_hard():
    env = EmailEnv()
    obs = env.reset()

    mistakes = 0

    while True:
        text = obs.text.lower()

        # intentionally harder logic
        if len(text) > 25:
            action = "mark_important"
        else:
            action = "mark_spam"

        obs, reward, done, _ = env.step(Action(action=action))

        if reward.reward < 0:
            mistakes += 1

        if done:
            break

    return max(0, 1 - (mistakes / 5))
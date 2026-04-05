import random
from env import EmailEnv

ACTIONS = ["spam", "important", "social"]

def run_inference(input_text: str):
    if not input_text.strip():
        return {
            "action": None,
            "reward": None,
            "message": "Empty email"
        }

    env = EmailEnv()
    env.reset(input_text)

    # -------------------------
    # PURE RANDOM ACTION (every click different)
    # -------------------------
    action = random.choice(ACTIONS)

    result = env.step(action)

    return {
        "action": action,
        "reward": result["reward"]
    }
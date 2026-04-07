import random
from env import EmailEnv

ACTIONS = ["spam", "important", "social"]


def run_inference(input_text: str):
    env = EmailEnv()
    state = env.reset(input_text)

    total_reward = 0
    steps = 1  # single-step env

    # -------------------------
    # START BLOCK
    # -------------------------
    print(f"[START] task=email_classification input={state}", flush=True)

    # -------------------------
    # STEP
    # -------------------------
    action = random.choice(ACTIONS)
    result = env.step(action)

    reward = result["reward"]
    total_reward += reward

    print(f"[STEP] step=1 action={action} reward={reward}", flush=True)

    # -------------------------
    # END BLOCK
    # -------------------------
    score = float(total_reward) # simple score

    print(
        f"[END] task=email_classification score={score} steps={steps}",
        flush=True
    )

    return {
        "action": action,
        "reward": reward
    }


# -------------------------
# MAIN (VERY IMPORTANT)
# -------------------------
if __name__ == "__main__":
    # sample run (validator triggers this)
    sample_email = "Win a lottery now!!!"
    run_inference(sample_email)
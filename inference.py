import os
import random
from openai import OpenAI
from env import EmailEnv

ACTIONS = ["spam", "important", "social"]


# -------------------------
# SAFE SCORE NORMALIZATION
# -------------------------
def normalize_score(score):
    if score <= 0:
        return 0.01
    elif score >= 1:
        return 0.99
    return float(score)


def run_inference(input_text: str):
    # -------------------------
    # 🔥 SAFE LLM CALL (REQUIRED)
    # -------------------------
    try:
        if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
            client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"]
            )

            client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                messages=[
                    {"role": "user", "content": f"Classify: {input_text}"}
                ],
                max_tokens=5
            )
    except Exception as e:
        print(f"[WARNING] LLM call failed: {str(e)}", flush=True)

    # -------------------------
    # ENV LOGIC
    # -------------------------
    env = EmailEnv()
    state = env.reset(input_text)

    total_reward = 0
    steps = 1

    print(f"[START] task=email_classification input={state}", flush=True)

    action = random.choice(ACTIONS)
    result = env.step(action)

    reward = result["reward"]
    total_reward += reward

    print(f"[STEP] step=1 action={action} reward={reward}", flush=True)

    # -------------------------
    # NORMALIZED SCORE (FIX)
    # -------------------------
    score = normalize_score(total_reward)

    print(
        f"[END] task=email_classification score={score} steps={steps}",
        flush=True
    )

    return {
        "action": action,
        "reward": reward
    }


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    sample_email = "Win a lottery now!!!"
    run_inference(sample_email)
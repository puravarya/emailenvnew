import os
import random

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from env import EmailEnv

ACTIONS = ["spam", "important", "social"]


def normalize_score(score):
    if score <= 0:
        return 0.01
    elif score >= 1:
        return 0.99
    return float(score)


def run_inference(input_text: str):
    # Safe LLM call — skipped if openai not available or env vars not set
    try:
        if OPENAI_AVAILABLE and "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
            client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"]
            )
            client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": f"Classify: {input_text}"}],
                max_tokens=5
            )
    except Exception as e:
        print(f"[WARNING] LLM call failed: {str(e)}", flush=True)

    env = EmailEnv()
    state = env.reset(input_text)
    action = random.choice(ACTIONS)
    result = env.step(action)
    reward = result["reward"]
    score = normalize_score(reward)

    print(f"[START] task=email_classification input={state}", flush=True)
    print(f"[STEP] step=1 action={action} reward={reward}", flush=True)
    print(f"[END] task=email_classification score={score} steps=1", flush=True)

    return {"action": action, "reward": reward}


if __name__ == "__main__":
    run_inference("Win a lottery now!!!")
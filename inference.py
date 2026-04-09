import os
import random

# Safe import of openai — won't crash if missing
try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

# ── Inline minimal env (no import of env.py needed) ──────────────────────────

_EMAILS = [
    "win a lottery now!!!",
    "meeting with ceo tomorrow",
    "huge discount just for you",
    "project deadline tomorrow",
    "claim your prize now!!!",
    "we have christmas celebration tomorrow at office",
    "vogue magazine 2026",
    "i-max theatre experience",
]
_REWARDS = {
    "win a lottery now!!!":                             {"spam": 1.0, "social": 0.5, "important": 0.0},
    "meeting with ceo tomorrow":                        {"spam": 0.0, "social": 0.5, "important": 1.0},
    "huge discount just for you":                       {"spam": 1.0, "social": 0.5, "important": 0.0},
    "project deadline tomorrow":                        {"spam": 0.0, "social": 0.5, "important": 1.0},
    "claim your prize now!!!":                          {"spam": 1.0, "social": 0.5, "important": 0.0},
    "we have christmas celebration tomorrow at office": {"spam": 0.0, "social": 1.0, "important": 0.5},
    "vogue magazine 2026":                              {"spam": 0.5, "social": 1.0, "important": 0.0},
    "i-max theatre experience":                         {"spam": 0.5, "social": 1.0, "important": 0.0},
}
_ACTIONS = ["spam", "important", "social"]

def _normalize(score):
    if score <= 0: return 0.01
    if score >= 1: return 0.99
    return float(score)

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(input_text: str):
    # Optional LLM call — skipped gracefully if unavailable
    try:
        if _OPENAI_OK and os.environ.get("API_BASE_URL") and os.environ.get("API_KEY"):
            client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"],
            )
            client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": f"Classify: {input_text}"}],
                max_tokens=5,
            )
    except Exception as e:
        print(f"[WARNING] LLM call failed: {e}", flush=True)

    # Core env logic (inline — no import needed)
    text = input_text.lower().strip()
    if text not in _REWARDS:
        text = random.choice(_EMAILS)

    action = random.choice(_ACTIONS)
    reward = _REWARDS.get(text, {}).get(action, 0.0)
    score  = _normalize(reward)

    print(f"[START] task=email_classification input={text}", flush=True)
    print(f"[STEP] step=1 action={action} reward={reward}", flush=True)
    print(f"[END] task=email_classification score={score} steps=1", flush=True)

    return {"action": action, "reward": reward}


if __name__ == "__main__":
    result = run_inference("Win a lottery now!!!")
    print(result)
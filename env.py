import random

class EmailEnv:
    def __init__(self):
        self.email_rewards = {
            "win a lottery now!!!": {
                "spam": 1, "social": 0.5, "important": -1
            },
            "meeting with ceo tomorrow": {
                "spam": -1, "social": 0.5, "important": 1
            },
            "huge discount just for you": {
                "spam": 1, "social": 0.5, "important": -1
            },
            "project deadline tomorrow": {
                "spam": -1, "social": 0.5, "important": 1
            },
            "claim your prize now!!!": {
                "spam": 1, "social": 0.5, "important": -1
            },
            "we have christmas celebration tomorrow at office": {
                "spam": -1, "social": 1, "important": 0.5
            },
            "vogue magazine 2026": {
                "spam": 0.5, "social": 1, "important": -1
            },
            "i-max theatre experience": {
                "spam": 0.5, "social": 1, "important": -1
            }
        }

        self.current_email = None

    def reset(self, input_text=None):
        if input_text:
            self.current_email = input_text.lower().strip()
        else:
            self.current_email = random.choice(list(self.email_rewards.keys()))
        return self.current_email

    def step(self, action):
        text = self.current_email

        # default reward if unknown email
        reward = 0

        if text in self.email_rewards:
            reward = self.email_rewards[text].get(action, 0)

        return {
            "observation": text,
            "reward": reward,
            "done": True
        }
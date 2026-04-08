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
        action = action.lower().replace("mark_", "")
        text = self.current_email
        reward = 0

        if text in self.email_rewards:
            reward = self.email_rewards[text].get(action, 0)

        return {
            "observation": text,
            "reward": reward,
            "done": True
        }


# -------------------------
# TASK 2: Spam Detector
# Binary classification: spam vs not_spam
# -------------------------
class SpamDetectorEnv:
    def __init__(self):
        self.emails = {
            "click here to win iphone": {"spam": 1, "not_spam": -1},
            "your invoice is attached": {"spam": -1, "not_spam": 1},
            "congratulations you won $1000": {"spam": 1, "not_spam": -1},
            "team standup at 10am": {"spam": -1, "not_spam": 1},
            "limited offer buy now": {"spam": 1, "not_spam": -1},
            "please review the attached report": {"spam": -1, "not_spam": 1},
            "you have been selected for a prize": {"spam": 1, "not_spam": -1},
            "quarterly review meeting invite": {"spam": -1, "not_spam": 1},
        }
        self.current_email = None

    def reset(self, input_text=None):
        if input_text:
            self.current_email = input_text.lower().strip()
        else:
            self.current_email = random.choice(list(self.emails.keys()))
        return self.current_email

    def step(self, action):
        action = action.lower().strip()
        text = self.current_email
        reward = 0

        if text in self.emails:
            reward = self.emails[text].get(action, 0)

        return {
            "observation": text,
            "reward": reward,
            "done": True
        }


# -------------------------
# TASK 3: Email Priority
# Priority levels: urgent, normal, low
# -------------------------
class EmailPriorityEnv:
    def __init__(self):
        self.emails = {
            "server is down production issue": {"urgent": 1, "normal": 0, "low": -1},
            "happy birthday wishes": {"urgent": -1, "normal": 0, "low": 1},
            "client contract needs signature today": {"urgent": 1, "normal": 0, "low": -1},
            "newsletter subscription confirmed": {"urgent": -1, "normal": 0, "low": 1},
            "critical bug in live system": {"urgent": 1, "normal": 0, "low": -1},
            "weekly team lunch reminder": {"urgent": -1, "normal": 0.5, "low": 1},
            "urgent approval needed for budget": {"urgent": 1, "normal": 0, "low": -1},
            "monthly analytics report": {"urgent": -1, "normal": 1, "low": 0},
        }
        self.current_email = None

    def reset(self, input_text=None):
        if input_text:
            self.current_email = input_text.lower().strip()
        else:
            self.current_email = random.choice(list(self.emails.keys()))
        return self.current_email

    def step(self, action):
        action = action.lower().strip()
        text = self.current_email
        reward = 0

        if text in self.emails:
            reward = self.emails[text].get(action, 0)

        return {
            "observation": text,
            "reward": reward,
            "done": True
        }
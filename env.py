import random


def _c(s):
    s = float(s)
    if s <= 0.0: return 0.01
    if s >= 1.0: return 0.99
    return round(s, 4)


class EmailEnv:
    def __init__(self):
        self.email_rewards = {
            "win a lottery now!!!":                             {"spam": 0.99, "social": 0.50, "important": 0.01},
            "meeting with ceo tomorrow":                        {"spam": 0.01, "social": 0.50, "important": 0.99},
            "huge discount just for you":                       {"spam": 0.99, "social": 0.50, "important": 0.01},
            "project deadline tomorrow":                        {"spam": 0.01, "social": 0.50, "important": 0.99},
            "claim your prize now!!!":                          {"spam": 0.99, "social": 0.50, "important": 0.01},
            "we have christmas celebration tomorrow at office": {"spam": 0.01, "social": 0.99, "important": 0.50},
            "vogue magazine 2026":                              {"spam": 0.50, "social": 0.99, "important": 0.01},
            "i-max theatre experience":                         {"spam": 0.50, "social": 0.99, "important": 0.01},
        }
        self.emails_list = list(self.email_rewards.keys())
        self.current_email = None

    def reset(self, input_text=None):
        if input_text:
            self.current_email = input_text.lower().strip()
        else:
            self.current_email = random.choice(self.emails_list)
        return self.current_email

    def step(self, action):
        action = action.lower().replace("mark_", "").strip()
        text = self.current_email
        raw = self.email_rewards.get(text, {}).get(action, 0.50)
        return {"observation": text, "reward": _c(raw), "done": True}


class SpamDetectorEnv:
    def __init__(self):
        self.emails = {
            "click here to win iphone":           {"spam": 0.99, "not_spam": 0.01},
            "your invoice is attached":           {"spam": 0.01, "not_spam": 0.99},
            "congratulations you won $1000":      {"spam": 0.99, "not_spam": 0.01},
            "team standup at 10am":               {"spam": 0.01, "not_spam": 0.99},
            "limited offer buy now":              {"spam": 0.99, "not_spam": 0.01},
            "please review the attached report":  {"spam": 0.01, "not_spam": 0.99},
            "you have been selected for a prize": {"spam": 0.99, "not_spam": 0.01},
            "quarterly review meeting invite":    {"spam": 0.01, "not_spam": 0.99},
        }
        self.emails_list = list(self.emails.keys())
        self.current_email = None

    def reset(self, input_text=None):
        if input_text:
            self.current_email = input_text.lower().strip()
        else:
            self.current_email = random.choice(self.emails_list)
        return self.current_email

    def step(self, action):
        action = action.lower().strip()
        text = self.current_email
        raw = self.emails.get(text, {}).get(action, 0.50)
        return {"observation": text, "reward": _c(raw), "done": True}


class EmailPriorityEnv:
    def __init__(self):
        self.emails = {
            "server is down production issue":       {"urgent": 0.99, "normal": 0.30, "low": 0.01},
            "happy birthday wishes":                 {"urgent": 0.01, "normal": 0.30, "low": 0.99},
            "client contract needs signature today": {"urgent": 0.99, "normal": 0.30, "low": 0.01},
            "newsletter subscription confirmed":     {"urgent": 0.01, "normal": 0.30, "low": 0.99},
            "critical bug in live system":           {"urgent": 0.99, "normal": 0.30, "low": 0.01},
            "weekly team lunch reminder":            {"urgent": 0.01, "normal": 0.50, "low": 0.99},
            "urgent approval needed for budget":     {"urgent": 0.99, "normal": 0.30, "low": 0.01},
            "monthly analytics report":              {"urgent": 0.01, "normal": 0.99, "low": 0.30},
        }
        self.emails_list = list(self.emails.keys())
        self.current_email = None

    def reset(self, input_text=None):
        if input_text:
            self.current_email = input_text.lower().strip()
        else:
            self.current_email = random.choice(self.emails_list)
        return self.current_email

    def step(self, action):
        action = action.lower().strip()
        text = self.current_email
        raw = self.emails.get(text, {}).get(action, 0.50)
        return {"observation": text, "reward": _c(raw), "done": True}
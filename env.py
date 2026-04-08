import random


class EmailEnv:
    """
    Task 1: Email Classification
    Classify emails as spam / important / social
    """
    def __init__(self):
        self.email_rewards = {
            "win a lottery now!!!": {
                "spam": 1.0, "social": 0.5, "important": 0.0
            },
            "meeting with ceo tomorrow": {
                "spam": 0.0, "social": 0.5, "important": 1.0
            },
            "huge discount just for you": {
                "spam": 1.0, "social": 0.5, "important": 0.0
            },
            "project deadline tomorrow": {
                "spam": 0.0, "social": 0.5, "important": 1.0
            },
            "claim your prize now!!!": {
                "spam": 1.0, "social": 0.5, "important": 0.0
            },
            "we have christmas celebration tomorrow at office": {
                "spam": 0.0, "social": 1.0, "important": 0.5
            },
            "vogue magazine 2026": {
                "spam": 0.5, "social": 1.0, "important": 0.0
            },
            "i-max theatre experience": {
                "spam": 0.5, "social": 1.0, "important": 0.0
            },
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
        reward = 0.0
        if text in self.email_rewards:
            reward = self.email_rewards[text].get(action, 0.0)
        return {"observation": text, "reward": reward, "done": True}


class SpamDetectorEnv:
    """
    Task 2: Spam Detection
    Binary classification: spam vs not_spam
    """
    def __init__(self):
        self.emails = {
            "click here to win iphone":           {"spam": 1.0, "not_spam": 0.0},
            "your invoice is attached":            {"spam": 0.0, "not_spam": 1.0},
            "congratulations you won $1000":       {"spam": 1.0, "not_spam": 0.0},
            "team standup at 10am":                {"spam": 0.0, "not_spam": 1.0},
            "limited offer buy now":               {"spam": 1.0, "not_spam": 0.0},
            "please review the attached report":   {"spam": 0.0, "not_spam": 1.0},
            "you have been selected for a prize":  {"spam": 1.0, "not_spam": 0.0},
            "quarterly review meeting invite":     {"spam": 0.0, "not_spam": 1.0},
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
        reward = 0.0
        if text in self.emails:
            reward = self.emails[text].get(action, 0.0)
        return {"observation": text, "reward": reward, "done": True}


class EmailPriorityEnv:
    """
    Task 3: Email Priority
    Three levels: urgent / normal / low
    """
    def __init__(self):
        self.emails = {
            "server is down production issue":       {"urgent": 1.0, "normal": 0.3, "low": 0.0},
            "happy birthday wishes":                 {"urgent": 0.0, "normal": 0.3, "low": 1.0},
            "client contract needs signature today": {"urgent": 1.0, "normal": 0.3, "low": 0.0},
            "newsletter subscription confirmed":     {"urgent": 0.0, "normal": 0.3, "low": 1.0},
            "critical bug in live system":           {"urgent": 1.0, "normal": 0.3, "low": 0.0},
            "weekly team lunch reminder":            {"urgent": 0.0, "normal": 0.5, "low": 1.0},
            "urgent approval needed for budget":     {"urgent": 1.0, "normal": 0.3, "low": 0.0},
            "monthly analytics report":              {"urgent": 0.0, "normal": 1.0, "low": 0.3},
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
        reward = 0.0
        if text in self.emails:
            reward = self.emails[text].get(action, 0.0)
        return {"observation": text, "reward": reward, "done": True}
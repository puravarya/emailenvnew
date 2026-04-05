---
title: EmailEnvNew
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📧 Email Triage RL Environment

This project simulates a **Reinforcement Learning (RL) environment** for email classification, where an agent learns to classify emails as **spam** or **important** based on reward feedback.

---

## 🚀 Live Demo

Interact with the system:
- Enter an email
- Get predicted action
- Receive reward based on correctness

---

## 🧠 Concept

This system mimics an RL setup:

- **State (Observation):** Email text  
- **Action:**  
  - `mark_spam`  
  - `mark_important`  
- **Reward:**  
  - ✅ +1 → correct classification  
  - 🟡 0.5 → partially correct (ambiguous case)  
  - ❌ -1 → incorrect classification  
  - ⚪ 0 → unknown / neutral  

The reward is **confidence-weighted**, meaning:
- Strong signals → higher reward
- Ambiguous emails → partial reward
- Wrong predictions → penalized

---

## ⚠️ Realistic Agent Behavior

Ambiguous emails may produce different actions to simulate real-world agent uncertainty.

Example:
- `"urgent prize claim"` → could be spam OR important
- The system reflects this via **partial rewards or variation**

---

## 📊 Features

- ✅ RL-style environment simulation  
- ✅ Confidence-based reward shaping  
- ✅ Handles ambiguous and noisy inputs  
- ✅ Interactive UI using Gradio  
- ✅ Batch evaluation with multiple scenarios  
- ✅ Deterministic grading logic (OpenEnv compliant)

---

## 🧪 Example Inputs

Try these:

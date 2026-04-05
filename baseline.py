from inference import run_inference

def run_baseline():
    test_cases = [
        "Win lottery now",
        "CEO meeting tomorrow",
        "urgent prize claim",
        "hello friend",
        "project deadline urgent"
    ]

    total_reward = 0

    print("=== BASELINE EVALUATION ===")

    for text in test_cases:
        result = run_inference(text)
        reward = result["reward"]
        total_reward += reward

        print(f"Input: {text}")
        print(f"Action: {result['action']}, Reward: {reward}")
        print("-" * 30)

    avg_score = total_reward / len(test_cases)

    print(f"\nAverage Score: {round(avg_score, 2)}")


if __name__ == "__main__":
    run_baseline()
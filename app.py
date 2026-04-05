import gradio as gr
from inference import run_inference

# Global session reward
total_reward = 0

def classify_email(input_text):
    global total_reward

    if not input_text.strip():
        return "⚠️ No email entered → No reward generated", total_reward

    result = run_inference(input_text)

    reward = result["reward"]
    total_reward += reward

    return f"""
📌 Predicted Action: {result['action']}
🎯 Reward: {reward}
""", total_reward


def reset_session():
    global total_reward
    total_reward = 0
    return "Session reset ✅", 0


with gr.Blocks() as demo:
    gr.Markdown("# 📧 Email Classification RL Environment")

    gr.Markdown(
        "This system mimics an RL environment where actions like spam/important classification generate rewards, helping train intelligent email agents."
    )

    # Input
    input_box = gr.Textbox(placeholder="Type email here...")

    # Outputs
    output_box = gr.Textbox(label="Result")
    total_box = gr.Number(label="🧮 Total Reward (Session)", value=0)

    # Buttons
    btn = gr.Button("Classify Email")
    reset_btn = gr.Button("Reset Session")

    # Actions
    btn.click(fn=classify_email, inputs=input_box, outputs=[output_box, total_box])
    reset_btn.click(fn=reset_session, outputs=[output_box, total_box])

    # Examples
    gr.Markdown("### 💡 Example Emails:")
    gr.Markdown("""
- Win a lottery now!!!
- Meeting with CEO tomorrow
- Huge discount just for you
- Project deadline tomorrow
- Claim your prize now!!!
- We have Christmas celebration tomorrow at office
- Vogue Magazine 2026
- I-max theatre experience
""")

demo.launch(server_name="0.0.0.0", server_port=7860)
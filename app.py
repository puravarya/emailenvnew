import gradio as gr
from inference import run_inference

def classify_email(input_text):
    if not input_text.strip():
        return "⚠️ No email entered → No reward generated"

    result = run_inference(input_text)

    return f"""
📌 Predicted Action: {result['action']}
🎯 Reward: {result['reward']}
"""

with gr.Blocks() as demo:
    gr.Markdown("# 📧 Email Classification RL Environment")

    gr.Markdown(
        "Each click generates a RANDOM classification (spam / important / social) and reward based on predefined rules."
    )

    gr.Markdown("### ✍️ Try your own email:")
    input_box = gr.Textbox(placeholder="Type email here...")

    output_box = gr.Textbox(label="Result")

    btn = gr.Button("Classify Email")
    btn.click(fn=classify_email, inputs=input_box, outputs=output_box)

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
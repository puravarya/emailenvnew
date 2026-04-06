from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from inference import run_inference

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Email Env API is running 🚀"}

@app.get("/test")
def test():
    return {"status": "working"}

@app.get("/web", response_class=HTMLResponse)
def web():
    return """
    <html>
        <body>
            <h2>Email Classifier 🚀</h2>
            <form action="/predict" method="post">
                <input name="input_text" placeholder="Enter email text" />
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict")
def predict(input_text: str = Form(...)):
    result = run_inference(input_text)
    return {"result": result}
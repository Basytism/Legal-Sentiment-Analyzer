from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Load HuggingFace sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

# Define request schema
class TextData(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Allow frontend to connect (localhost / Colab / hosted)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:5500"] for local HTML
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
def analyze(data: TextData):
    # Split long text into sentences
    lines = [line.strip() for line in data.text.splitlines() if len(line.strip()) > 30]
    predictions = sentiment_pipeline(lines[:50])  # limit to 50 for speed
    result = [{"text": line, "label": pred["label"]} for line, pred in zip(lines, predictions)]
    return {"results": result}

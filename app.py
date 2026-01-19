from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI(title="CogniShield Tier-1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Load models at startup
# ==========================

# ✅ AI-generated text model (Transformer)
MODEL_NAME = "distilroberta-base"  # or the model you trained with

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ai_text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
ai_text_model.eval()

# Fake news ANN model
fake_news_model = load_model("models/fake_news_ann.h5")

# TF-IDF vectorizer
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ==========================
# Request schema
# ==========================
class TextRequest(BaseModel):
    text: str

# ==========================
# Health check
# ==========================
@app.get("/")
def home():
    return {"status": "running", "message": "CogniShield API is live"}

# ==========================
# Prediction endpoint
# ==========================
@app.post("/analyze")
def analyze_text(request: TextRequest):

    text = request.text.strip()[:2000]

    # --------------------------
    # Fake News Detection (ANN)
    # --------------------------
    X_tfidf = tfidf_vectorizer.transform([text]).toarray()

    probs_fake = fake_news_model.predict(X_tfidf)[0]
    fake_confidence = float(probs_fake[1])
    fake_news = fake_confidence > 0.5

    # --------------------------
    # AI-generated Detection (Transformer)
    # --------------------------
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = ai_text_model(**inputs)
        probs_ai = F.softmax(outputs.logits, dim=1)

    ai_confidence = probs_ai[0][1].item()
    ai_generated = ai_confidence > 0.7  # adjustable threshold

    # --------------------------
    # Response
    # --------------------------
    return {
        "ai_generated": ai_generated,
        "ai_confidence": round(ai_confidence, 4),
        "fake_news": fake_news,
        "fake_confidence": round(fake_confidence, 4)
    }

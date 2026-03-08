from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import os

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0"
)

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Mount Static Files
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Templates Directory
# -----------------------------
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Model Paths
# -----------------------------
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

# -----------------------------
# Load Model on Startup
# -----------------------------
@app.on_event("startup")
def load_model():
    global model, scaler

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Train the model first.")

    if not os.path.exists(SCALER_PATH):
        raise RuntimeError("Scaler file not found. Run preprocessing first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Home Page (Frontend)
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Input Schema
# -----------------------------
class CustomerData(BaseModel):
    features: list[float]

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerData):

    try:
        features = np.array(data.features).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict probability
        probability = model.predict_proba(features_scaled)[0][1]

        # Convert to label
        prediction = "Yes" if probability > 0.5 else "No"

        return {
            "prediction": prediction,
            "churn_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
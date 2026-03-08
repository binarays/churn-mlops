from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import os
import logging

logger = logging.getLogger("uvicorn.error")

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
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or scaler: {e}")

# -----------------------------
# Home Page (Frontend)
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
def health():
    if model is None or scaler is None:
        return {"status": "unhealthy"}
    return {"status": "healthy"}

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
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = "Yes" if probability > 0.5 else "No"
        logger.info(f"Prediction: {prediction}, probability: {probability:.2f}")
        return {"prediction": prediction, "churn_probability": float(probability)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
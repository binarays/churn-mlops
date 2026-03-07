from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
# Enable CORS (important for frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model Paths
# -----------------------------
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# -----------------------------
# Load Model Safely
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Please train the model first.")

if not os.path.exists(SCALER_PATH):
    raise RuntimeError("Scaler file not found. Run preprocessing first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Input Schema
# -----------------------------
class CustomerData(BaseModel):
    features: list[float]

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerData):

    try:
        # Convert to numpy array
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
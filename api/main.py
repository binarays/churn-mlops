from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0"
)


# -----------------------------
# Load Model and Scaler
# -----------------------------
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Train the model first.")

if not os.path.exists(SCALER_PATH):
    raise RuntimeError("Scaler file not found. Run preprocessing first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# -----------------------------
# Input Schema
# -----------------------------
class CustomerData(BaseModel):
    features: list


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerData):

    try:
        features = np.array(data.features).reshape(1, -1)

        # scale input
        features_scaled = scaler.transform(features)

        probability = model.predict_proba(features_scaled)[0][1]

        prediction = "Yes" if probability > 0.5 else "No"

        return {
            "churn_probability": float(probability),
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
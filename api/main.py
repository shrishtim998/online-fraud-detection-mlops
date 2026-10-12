
from pathlib import Path
import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "final_fraud_pipeline.pkl"
THRESHOLD_PATH = REPORT_DIR / "best_threshold.json"

# FastAPI application
app = FastAPI(
    title="Online Fraud Detection API",
    description="Predict whether a financial transaction is fraudulent.",
    version="1.0.0",
)

# Load the trained pipeline once when the API starts
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Load the configured decision threshold
threshold = 0.5

if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)
        threshold = float(threshold_data.get("threshold", 0.5))

if not 0 <= threshold <= 1:
    raise ValueError("The decision threshold must be between 0 and 1.")


# Request schema and input validation
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "DEBIT",
        "CASH_IN",
    ]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


# Response schema
class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    threshold: float
    label: Literal["Fraud", "Legitimate"]


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Online Fraud Detection API is running",
        "model": "Final Fraud Detection Pipeline",
        "threshold": threshold,
    }


# Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# Fraud prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        # Convert validated request into a DataFrame
        row = pd.DataFrame([transaction.model_dump()])

        # Calculate the features used during model training
        row["origin_balance_discrepancy"] = (
            row["oldbalanceOrg"]
            - row["amount"]
            - row["newbalanceOrig"]
        )

        row["destination_balance_discrepancy"] = (
            row["oldbalanceDest"]
            + row["amount"]
            - row["newbalanceDest"]
        )

        # The trained pipeline handles preprocessing and encoding
        probability = float(model.predict_proba(row)[0, 1])

        # Convert probability into a class prediction
        prediction = int(probability >= threshold)

        return {
            "is_fraud": prediction,
            "fraud_probability": round(probability, 6),
            "threshold": threshold,
            "label": "Fraud" if prediction else "Legitimate",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

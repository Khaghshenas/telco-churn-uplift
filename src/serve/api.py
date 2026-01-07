import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

MODEL_PATH = "models/churn_model.joblib"
FEATURE_COLUMNS_PATH = "data/processed/feature_columns.json"

app = FastAPI(title="Telco Churn Prediction API")


# ---------- Load model + feature columns ----------

model = joblib.load(MODEL_PATH)

with open(FEATURE_COLUMNS_PATH, "r") as f:
    feature_columns = json.load(f)


# ---------- Request Schema ----------

class CustomerFeatures(BaseModel):
    features: dict   # raw feature input from client


# ---------- Internal helper: preprocess input ----------

def preprocess_input(features: dict) -> pd.DataFrame:
    """
    Convert incoming feature dict into a DataFrame,
    perform one-hot encoding, and align with training columns.
    """
    df = pd.DataFrame([features])

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns from training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns and ordering
    df = df[feature_columns]

    return df


# ---------- Main Prediction Endpoint ----------

@app.post("/predict")
def predict(request: CustomerFeatures):
    df = preprocess_input(request.features)
    proba = model.predict_proba(df)[0, 1]
    
    return {
        "churn_probability": float(proba)
    }


# ---------- Root Endpoint ----------

@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is running."}

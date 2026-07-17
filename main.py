"""
FastAPI application for Fraud Detection.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Optional, List

app = FastAPI(
    title="Fraud Detection API",
    description="API for credit card fraud detection",
    version="1.0.0"
)

# Request model
class PredictionRequest(BaseModel):
    features: List[float]
    transaction_id: Optional[str] = None

# Response model
class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    prediction: int
    probability: float
    status: str

# Load model
MODEL_PATH = "models/production_model.pkl"
SCALER_PATH = "models/scaler.pkl"
IMPUTER_PATH = "models/imputer.pkl"

model = None
scaler = None
imputer = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"⚠️ Scaler not loaded: {e}")

try:
    if os.path.exists(IMPUTER_PATH):
        imputer = joblib.load(IMPUTER_PATH)
        print(f"✅ Imputer loaded from {IMPUTER_PATH}")
except Exception as e:
    print(f"⚠️ Imputer not loaded: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict if a transaction is fraudulent.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Apply preprocessing if scaler exists
        if scaler:
            features_df = pd.DataFrame(
                scaler.transform(features_df),
                columns=features_df.columns
            )
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0].max()
        
        return PredictionResponse(
            transaction_id=request.transaction_id,
            prediction=int(prediction),
            probability=float(probability),
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
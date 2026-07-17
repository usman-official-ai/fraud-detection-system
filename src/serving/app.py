from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import logging
from typing import List, Optional
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection",
    version="1.0.0"
)

# Load models and transformers
try:
    model = joblib.load('models/production_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    logger.info("Model, scaler, and imputer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    scaler = None
    imputer = None

class TransactionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    processing_time_ms: float

def engineer_features(df):
    """Apply the same feature engineering as training"""
    df_copy = df.copy()
    
    # Time-based features
    if 'Time' in df_copy.columns:
        df_copy['hour'] = df_copy['Time'] // 3600 % 24
        df_copy['day'] = df_copy['Time'] // (3600 * 24)
    
    # Amount features
    if 'Amount' in df_copy.columns:
        df_copy['amount_log'] = np.log1p(df_copy['Amount'])
    
    # Interaction features
    if 'V1' in df_copy.columns and 'V2' in df_copy.columns:
        df_copy['V1_V2_interaction'] = df_copy['V1'] * df_copy['V2']
    
    if 'V3' in df_copy.columns and 'V4' in df_copy.columns:
        df_copy['V3_V4_interaction'] = df_copy['V3'] * df_copy['V4']
    
    if 'Amount' in df_copy.columns and 'V1' in df_copy.columns:
        df_copy['amount_v1_ratio'] = df_copy['Amount'] / (abs(df_copy['V1']) + 1)
    
    return df_copy

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "status": "healthy" if model else "unhealthy",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    if model is None or scaler is None or imputer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([request.dict()])
        
        # Apply feature engineering
        df_engineered = engineer_features(df)
        
        # Get numeric columns for imputation
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply imputation
        df_imputed = df_engineered.copy()
        df_imputed[numeric_cols] = imputer.transform(df_engineered[numeric_cols])
        
        # Apply scaling
        df_scaled = df_imputed.copy()
        df_scaled[numeric_cols] = scaler.transform(df_imputed[numeric_cols])
        
        # Predict
        probability = model.predict_proba(df_scaled)[0, 1]
        prediction = int(probability >= 0.5)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_score=float(probability * 100),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(transactions: List[TransactionRequest]):
    if model is None or scaler is None or imputer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions])
        
        # Apply feature engineering
        df_engineered = engineer_features(df)
        
        # Get numeric columns for imputation
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply imputation
        df_imputed = df_engineered.copy()
        df_imputed[numeric_cols] = imputer.transform(df_engineered[numeric_cols])
        
        # Apply scaling
        df_scaled = df_imputed.copy()
        df_scaled[numeric_cols] = scaler.transform(df_imputed[numeric_cols])
        
        # Predict
        probabilities = model.predict_proba(df_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "processing_time_ms": round(processing_time, 2),
            "count": len(transactions)
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "imputer_loaded": imputer is not None
    }
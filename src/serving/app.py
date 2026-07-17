from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import logging
from typing import List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model
try:
    model = joblib.load('models/production_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    logger.info("✅ Model loaded successfully")
    logger.info(f"Model expects {model.n_features_in_} features")
    logger.info(f"Scaler expects {scaler.n_features_in_} features")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    scaler = None
    imputer = None

class TransactionRequest(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(request: TransactionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    start = time.time()
    
    try:
        # Step 1: Input features
        logger.info(f"Input features count: {len(request.features)}")
        
        # Step 2: Create DataFrame
        cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                'V28', 'Amount']
        
        df = pd.DataFrame([request.features], columns=cols)
        logger.info(f"DataFrame shape after creation: {df.shape}")
        
        # Step 3: Feature engineering
        df_copy = df.copy()
        
        # Time features
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
        
        # Ensure 36 columns
        final_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
            'V28', 'Amount', 'hour', 'day', 'amount_log', 
            'V1_V2_interaction', 'V3_V4_interaction', 'amount_v1_ratio'
        ]
        
        for col in final_columns:
            if col not in df_copy.columns:
                df_copy[col] = 0.0
        
        df_eng = df_copy[final_columns]
        logger.info(f"After feature engineering shape: {df_eng.shape}")
        logger.info(f"Columns: {df_eng.columns.tolist()}")
        
        # Step 4: Impute
        df_imp = pd.DataFrame(
            imputer.transform(df_eng),
            columns=df_eng.columns
        )
        logger.info(f"After imputation shape: {df_imp.shape}")
        
        # Step 5: Scale
        df_scaled = pd.DataFrame(
            scaler.transform(df_imp),
            columns=df_imp.columns
        )
        logger.info(f"After scaling shape: {df_scaled.shape}")
        
        # Step 6: Predict
        prob = model.predict_proba(df_scaled)[0, 1]
        pred = int(prob >= 0.5)
        
        return {
            "is_fraud": bool(pred),
            "fraud_probability": float(prob),
            "risk_score": float(prob * 100),
            "processing_time_ms": round((time.time() - start) * 1000, 2)
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
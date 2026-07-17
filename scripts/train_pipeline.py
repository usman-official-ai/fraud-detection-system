import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
import logging
from sklearn.metrics import classification_report

from src.data_pipeline.ingestion import DataIngestion
from src.features.engineering import FeatureEngineer
from src.models.train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Starting training pipeline...")
    
    # 1. Load and split data
    ingestion = DataIngestion(data_path='data/')
    df = ingestion.load_data()
    
    splits = ingestion.split_data(df)
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']
    
    # 2. Feature engineering
    engineer = FeatureEngineer(scaler_type='robust')
    
    X_train_engineered = engineer.create_features(X_train)
    X_val_engineered = engineer.create_features(X_val)
    X_test_engineered = engineer.create_features(X_test)
    
    # 3. Scale features (includes imputation)
    X_train_scaled = engineer.scale_features(X_train_engineered)
    X_val_scaled = engineer.scale_features(X_val_engineered)
    X_test_scaled = engineer.scale_features(X_test_engineered)
    
    # 4. Verify no NaN values remain
    assert not np.isnan(X_train_scaled).any().any(), "NaN values found in training data"
    assert not np.isnan(X_val_scaled).any().any(), "NaN values found in validation data"
    
    logger.info("Data validation passed - no NaN values")
    
    # 5. Train models
    trainer = ModelTrainer()
    results, best_model = trainer.train_all_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val
    )
    
    # 6. Test best model
    y_pred_test = best_model.predict(X_test_scaled)
    
    logger.info("\nTest Results:")
    logger.info(classification_report(y_test, y_pred_test))
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
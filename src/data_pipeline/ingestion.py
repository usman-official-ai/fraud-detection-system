import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_data(self):
        """Load Credit Card Fraud Dataset"""
        df = pd.read_csv(self.data_path / 'creditcard.csv')
        logger.info(f"Loaded {len(df)} transactions")
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Split into train/val/test"""
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
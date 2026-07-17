import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

class FeatureEngineer:
    def __init__(self, scaler_type='robust'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def create_features(self, df):
        """Create additional features"""
        df_copy = df.copy()
        
        # Time-based features
        if 'Time' in df_copy.columns:
            df_copy['hour'] = df_copy['Time'] // 3600 % 24
            df_copy['day'] = df_copy['Time'] // (3600 * 24)
        
        # Amount features - convert to numeric, not categorical
        if 'Amount' in df_copy.columns:
            df_copy['amount_log'] = np.log1p(df_copy['Amount'])
            # Remove categorical binning - it's causing issues
            # df_copy['amount_bin'] = pd.cut(...)  # REMOVED
        
        # Interaction features
        v_features = [f'V{i}' for i in range(1, 29) if f'V{i}' in df_copy.columns]
        
        if 'V1' in df_copy.columns and 'V2' in df_copy.columns:
            df_copy['V1_V2_interaction'] = df_copy['V1'] * df_copy['V2']
        
        if 'V3' in df_copy.columns and 'V4' in df_copy.columns:
            df_copy['V3_V4_interaction'] = df_copy['V3'] * df_copy['V4']
        
        if 'Amount' in df_copy.columns and 'V1' in df_copy.columns:
            df_copy['amount_v1_ratio'] = df_copy['Amount'] / (abs(df_copy['V1']) + 1)
        
        # Ensure all columns are numeric
        for col in df_copy.columns:
            if df_copy[col].dtype == 'category':
                df_copy[col] = df_copy[col].astype(float)
        
        self.feature_columns = [col for col in df_copy.columns if col not in ['Class']]
        
        return df_copy
    
    def scale_features(self, df):
        """Scale numeric features and handle missing values"""
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Initialize imputer if not exists
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='median')
            self.imputer.fit(df[numeric_cols])
            joblib.dump(self.imputer, 'models/imputer.pkl')
        
        # Impute missing values
        df_imputed = df.copy()
        df_imputed[numeric_cols] = self.imputer.transform(df[numeric_cols])
        
        # Scale features
        if self.scaler is None:
            if self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            self.scaler.fit(df_imputed[numeric_cols])
            joblib.dump(self.scaler, 'models/scaler.pkl')
        
        df_scaled = df_imputed.copy()
        df_scaled[numeric_cols] = self.scaler.transform(df_imputed[numeric_cols])
        
        return df_scaled
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_feature_matrix(df):
    """
    清洗特征矩阵：处理无穷大值和缺失值。
    Ref:  Handling missing values
    """
    df_clean = df.copy()
    
    # replace inf with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # check for NaN values
    nan_count = df_clean.isna().sum().sum()
    if nan_count > 0:
        print(f"[Preprocessing] Found {nan_count} missing values. Filling with column mean...")
        # fill with column mean (only for numeric columns)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
    return df_clean

def normalize_features(df, feature_cols, method='z-score'):

    df_scaled = df.copy()
    
    if method == 'z-score':
        scaler = StandardScaler()
    elif method == 'min-max':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'z-score' or 'min-max'")
        
    # Fit and transform
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"[Preprocessing] Features normalized using {method}.")
    return df_scaled, scaler
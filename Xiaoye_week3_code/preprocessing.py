import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_feature_matrix(df):
    """
    Ref:  Handling missing values
    """
    df_clean = df.copy()
    
    # replace inf with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # check for NaN values
    nan_count = df_clean.isna().sum().sum()
    if nan_count > 0:
        print(f"[Preprocessing] Found {nan_count} missing values. Filling with column mean...")
        
        # using linear interpolation for numeric columns, then fill remaining NaNs with column mean
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear', limit_direction='both')
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
    return df_clean


def remove_outliers(df, feature_cols, factor=3.0): 
    """
    Removes outliers using the IQR (Interquartile Range) method.
    Ref: Removing outliers (IQR-based).
    
    Args:
        factor: The strictness of the filter. 
                1.5 is standard, but for Fall Detection, 3.0 (extreme outliers) is safer 
                to avoid deleting actual fall impacts.
    """
    df_clean = df.copy()
    original_len = len(df_clean)
    
    # Only remove outliers from these columns
    for col in feature_cols:
        if col not in df_clean.columns: continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Filtering
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
    removed_count = original_len - len(df_clean)
    if removed_count > 0:
        print(f"[Preprocessing] Outliers removed: {removed_count} rows (Factor={factor}).")
        
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
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import butter, filtfilt

# Modified from analysis_features.py to be used in server.py for real-time feature extraction


def compute_spectral_features(signal, fs=50):
  
    if len(signal) < 2:
        return 0, 0
    
   
    windowed = signal * np.hamming(len(signal))
    fft_vals = np.fft.rfft(windowed)
    psd = np.abs(fft_vals) ** 2
    spectral_energy = np.sum(psd)
    
    psd_norm = psd / (np.sum(psd) + 1e-9) 
    spectral_ent = entropy(psd_norm, base=2)
    
    return spectral_energy, spectral_ent



def preprocess_window(window_data, fs=50, cutoff=15, order=4):
 
    if isinstance(window_data, list):
        df = pd.DataFrame(window_data)
    else:
        df = window_data.copy()
        

    required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for c in required_cols:
        if c not in df.columns: df[c] = 0.0
        

    try:

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        if len(df) > 3 * (order + 1):
            for col in required_cols:
                df[col] = filtfilt(b, a, df[col].values)
    except Exception as e:
        pass

  
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    
    return df


def extract_features_from_window(w, fs=50):
  
    
    acc = w["acc_mag"].values
    gyro = w["gyro_mag"].values
    
    feats = {}

    # time domain features
    feats["acc_max"] = np.max(acc)
    feats["acc_mean"] = np.mean(acc)
    feats["acc_std"] = np.std(acc)
    feats["gyro_max"] = np.max(gyro)
    feats["gyro_mean"] = np.mean(gyro)
    feats["gyro_std"] = np.std(gyro)
    feats["acc_skew"] = skew(acc)
    feats["acc_kurtosis"] = kurtosis(acc)

    # frequency domain features
    feats["acc_spec_energy"], feats["acc_spec_entropy"] = compute_spectral_features(acc, fs)
    feats["gyro_spec_energy"], feats["gyro_spec_entropy"] = compute_spectral_features(gyro, fs)

    return feats
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy import stats
from scipy.signal import butter, filtfilt


# smoothing data - butterworth low-pass filter
def apply_lowpass_filter(data_dict, cutoff=15, fs=50, order=4):
    """
    zero-phase Butterworth low-pass filter applied to each sensor axis, grouped by 'seg_id' to prevent cross-gap smoothing.
    
    Args:
       cutoff: Desired cutoff frequency of the filter (Hz). For human motion, 15Hz is a common choice.
    """
    smoothed_dict = copy.deepcopy(data_dict)
    
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    padlen = 3 * (order + 1)

    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    # print(f"Applying Butterworth Low-Pass (Cutoff={cutoff}Hz) respecting seg_id...")
    
    for fname, df in smoothed_dict.items():
        # safety check: if 'seg_id' doesn't exist, create a default one (treat whole file as one segment)
        if 'seg_id' not in df.columns:
            df['seg_id'] = 0
            
        # smoothing each axis 
        for col in sensor_cols:
            if col in df.columns:
                # only filter if we have enough data points in the segment to avoid edge artifacts
                def filter_segment(series):
                    if len(series) > padlen: 
                        return filtfilt(b, a, series.values)
                    else:
                        return series.values 
                
                # [guarantee no cross-segment smoothing] Apply filter within each segment defined by 'seg_id'
                df[col] = df.groupby('seg_id')[col].transform(filter_segment)
        
        # recalculate magnitudes after smoothing
        if set(['ax', 'ay', 'az']).issubset(df.columns):
            df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
            
        if set(['gx', 'gy', 'gz']).issubset(df.columns):
            df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
            
    # print("Smoothing & Recalculation complete.")
    return smoothed_dict


# sliding window slicing
def sliding_windows(df, window_s=2.0, step_s=1.0, fs=50):
    """
    slicing window based on time (t_s) and segment id (seg_id) to ensure windows do not cross gaps.
    """
    local_df = df.copy()
    
    if 'seg_id' not in local_df.columns:
        local_df['seg_id'] = 0
    
    win_n = int(window_s * fs)
    step_n = int(step_s * fs)
    windows = []


    groups = local_df.groupby("seg_id")

    for _, g in groups:
        g = g.reset_index(drop=True)
        n_samples = len(g)
        
        if n_samples < win_n:
            continue

        start = 0
        while start + win_n <= n_samples:
            w = g.iloc[start : start + win_n].copy()
            windows.append(w)
            start += step_n
            
    return windows

# Feature Extraction
def compute_spectral_features(signal, fs=50):
    """
    Computes frequency domain features: Spectral Energy and Spectral Entropy.
    """
    if len(signal) < 2:
        return 0, 0
    
    windowed = signal * np.hamming(len(signal))
    fft_vals = np.fft.rfft(windowed)
    psd = np.abs(fft_vals) ** 2
    spectral_energy = np.sum(psd)
    psd_norm = psd / (np.sum(psd) + 1e-9) 
    spectral_ent = entropy(psd_norm, base=2)
    return spectral_energy, spectral_ent


def extract_features_from_window(w, fs=50):
    """
    Calculate statistical and frequency features for a single window.
    """
    # Extract numpy arrays for efficiency
    acc = w["acc_mag"].values
    gyro = w["gyro_mag"].values
    
    feats = {}

    
    # time domain -  Amplitude
    feats["acc_max"] = np.max(acc)
    feats["acc_mean"] = np.mean(acc)
    feats["acc_std"] = np.std(acc)
    feats["gyro_max"] = np.max(gyro)
    feats["gyro_mean"] = np.mean(gyro)
    feats["gyro_std"] = np.std(gyro)

    # time domain -  Shape / Distribution (Critical for Falls)
    feats["acc_skew"] = skew(acc)
    feats["acc_kurtosis"] = kurtosis(acc)

    # frequency domain features
    feats["acc_spec_energy"], feats["acc_spec_entropy"] = compute_spectral_features(acc, fs)
    feats["gyro_spec_energy"], feats["gyro_spec_entropy"] = compute_spectral_features(gyro, fs)

    return feats



def build_feature_dataset(file_dict, window_s=2.0, step_s=1.0, fs=50):
    """
    Iterate over all files, slice windows, and extract features.
    """
    all_features = []

    FALL_RADIUS_S = 1.0
    
    for filename, df in file_dict.items():
        fname_lower = filename.lower()
        
        if "fall" in fname_lower:
            is_fall_file = True
            activity_type = "Fall"
            
            # peak
            peak_idx = df['acc_mag'].idxmax()
            peak_time = df.loc[peak_idx, 't_s']
            
        else:
            is_fall_file = False
            if "walk" in fname_lower: activity_type = "Walking"
            else: activity_type = "Sit-Stand"
            peak_time = -999 # Dummy value

        # 2. Sliding window slicing
        windows = sliding_windows(df, window_s=window_s, step_s=step_s, fs=fs)
        
        for w in windows:
            f_dict = extract_features_from_window(w, fs)
            
            # the center of the window
            w_center = (w['t_s'].iloc[0] + w['t_s'].iloc[-1]) / 2
            
            
            label = 0 
            
            if is_fall_file:
                if abs(w_center - peak_time) <= FALL_RADIUS_S:
                    label = 1
                else:
                    label = 0
            

            else:
                label = 0
            
            f_dict["file"] = filename
            f_dict["label"] = label
            f_dict["activity"] = activity_type

            f_dict["t_start"] = w["t_s"].iloc[0]
            f_dict["t_end"] = w["t_s"].iloc[-1]
            
            all_features.append(f_dict)
            
    return pd.DataFrame(all_features)





# statistical analysis
def get_statistical_report(df, feature_list, group_col="binary_class", target_label="Fall"):
    """
    Perform independent t-tests and compute summary statistics (Mean +/- Std)
    to compare two groups (e.g., Fall vs. Non-Fall).
   
    Args:
        df: DataFrame containing features and labels.
        feature_list: List of feature names to analyze.
        group_col: Column name used for grouping (must contain 2 distinct values).
        target_label: The label considered as the 'positive' class (e.g., 'Fall').
        
    Returns:
        pd.DataFrame: A formatted statistical report.
    """
    # Check if the grouping column exists
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    # Get unique labels
    labels = df[group_col].unique()
    if len(labels) != 2:
        raise ValueError(f"t-test requires exactly 2 groups. Found {len(labels)}: {labels}")

    # Identify the two groups
    # Group 1 = Target (Fall), Group 2 = The other one (Non-Fall)
    group_1 = df[df[group_col] == target_label]
    other_label = [l for l in labels if l != target_label][0]
    group_2 = df[df[group_col] == other_label]
    
    results = []
    
    for feat in feature_list:
        if feat not in df.columns:
            continue
            
        # Drop NaNs to ensure clean calculation
        v1 = group_1[feat].dropna()
        v2 = group_2[feat].dropna()
        
        # 1. Summary Statistics 
        mean1, std1 = v1.mean(), v1.std()
        mean2, std2 = v2.mean(), v2.std()
        
        # 2. Statistical Test (Welch's t-test) 
        # equal_var=False handles unequal sample sizes and variances (common in Fall data)
        t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
        
        # Determine significance stars
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        else: sig = "ns" # not significant
        
        results.append({
            "Feature": feat,
            f"{target_label} (Mean +/- Std)": f"{mean1:.2f} +/- {std1:.2f}",
            f"{other_label} (Mean +/- Std)": f"{mean2:.2f} +/- {std2:.2f}",
            "t-statistic": f"{t_stat:.2f}",
            "p-value": f"{p_val:.2e}",
            "Significance": sig
        })
        
    return pd.DataFrame(results)


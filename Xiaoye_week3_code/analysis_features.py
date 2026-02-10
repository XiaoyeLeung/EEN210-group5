import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from mpl_toolkits.mplot3d import Axes3D


# sliding window slicing
def sliding_windows(df, window_s=2.0, step_s=1.0, fs=50):
    """
    Slice the dataframe into overlapping windows.
    
    Args:
        df: Input dataframe (must contain 't_s' and sensor columns).
        window_s: Window size in seconds (e.g., 2.0s).
        step_s: Step size in seconds (e.g., 1.0s for 50% overlap).
        fs: Sampling frequency.
        
    Returns:
        List of DataFrame windows.
    """
    win_n = int(window_s * fs)
    step_n = int(step_s * fs)
    windows = []

    # Handle gaps: Do not slice across data gaps (is_gap == True)
    # If 'is_gap' exists, use it to split segments; otherwise treat as one segment
    if "is_gap" in df.columns and df["is_gap"].any():
        # Create a group ID that increments every time a gap is encountered
        df["seg_id"] = df["is_gap"].cumsum()
        groups = df.groupby("seg_id")
    else:
        groups = [(0, df)]

    for _, g in groups:
        g = g.reset_index(drop=True)
        n_samples = len(g)
        
        # Skip segments shorter than one window
        if n_samples < win_n:
            continue

        start = 0
        while start + win_n <= n_samples:
            w = g.iloc[start : start + win_n].copy()
            windows.append(w)
            start += step_n
            
    return windows

# Features

def compute_spectral_features(signal, fs=50):
    """
    Computes frequency domain features: Spectral Energy and Spectral Entropy.
    """
    if len(signal) < 2:
        return 0, 0
        
    # 1. Apply Hamming window to reduce spectral leakage
    windowed = signal * np.hamming(len(signal))
    
    # 2. Compute FFT (Real FFT since signal is real-valued)
    fft_vals = np.fft.rfft(windowed)
    
    # 3. Compute Power Spectral Density (PSD)
    # PSD represents the power distribution across frequencies
    psd = np.abs(fft_vals) ** 2
    
    # Spectral Energy
    spectral_energy = np.sum(psd)
    
    # Spectral Entropy 
    # Normalize PSD to treat it like a probability distribution
    psd_norm = psd / np.sum(psd)
    # Compute Shannon Entropy (scipy.stats.entropy uses ln by default, base=2 is common for bits)
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

    
    # Intensity / Amplitude
    feats["acc_max"] = np.max(acc)
    feats["acc_mean"] = np.mean(acc)
    feats["acc_var"] = np.var(acc)
    feats["acc_std"] = np.std(acc)
    
    feats["gyro_max"] = np.max(gyro)
    feats["gyro_mean"] = np.mean(gyro)
    feats["gyro_var"] = np.var(gyro)
    feats["gyro_std"] = np.std(gyro)

    # Shape / Distribution (Critical for Falls)
    # Skewness: Measures asymmetry. Falls often have high positive skew (one-sided spike).
    feats["acc_skew"] = skew(acc)
    # Kurtosis: Measures "tailedness" or impulsiveness. Impacts have very high kurtosis.
    feats["acc_kurtosis"] = kurtosis(acc) 
    

    # Frequency Domain Features 
    acc_spec_energy, acc_spec_entropy = compute_spectral_features(acc, fs)
    feats["acc_spec_energy"] = acc_spec_energy
    feats["acc_spec_entropy"] = acc_spec_entropy
    
    # Calculate for Gyroscope (Optional but recommended)
    gyro_spec_energy, gyro_spec_entropy = compute_spectral_features(gyro, fs)
    feats["gyro_spec_energy"] = gyro_spec_energy
    feats["gyro_spec_entropy"] = gyro_spec_entropy

    return feats


    return feats


def build_feature_dataset(file_dict, window_s=2.0, step_s=1.0, fs=50):
    """
    Iterate over all files, slice windows, and extract features.

    Args:
        file_dict: Dictionary {filename: DataFrame}
    """
    all_features = []
    
    for filename, df in file_dict.items():
        # Determine label based on filename (Simple rule-based)
        # 1 = Fall, 0 = Non-Fall (ADL)
        fname_lower = filename.lower()
        if "fall" in fname_lower:
            label = 1
            activity_type = "Fall"
        elif "walk" in fname_lower:
            label = 0
            activity_type = "Walking"
        elif "sit" in fname_lower or "stand" in fname_lower:
            label = 0
            activity_type = "Sit-Stand"
        else:
            label = 0
            activity_type = "Other"

        # Sliding Window
        windows = sliding_windows(df, window_s=window_s, step_s=step_s, fs=fs)
        
        for w in windows:
            # Extract
            f_dict = extract_features_from_window(w, fs)
            
            # Add Metadata
            f_dict["file"] = filename
            f_dict["label"] = label
            f_dict["activity"] = activity_type
            f_dict["t_start"] = w["t_s"].iloc[0]
            f_dict["t_end"] = w["t_s"].iloc[-1]
            
            all_features.append(f_dict)
            
    return pd.DataFrame(all_features)












# statistical analysis
from scipy import stats

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

def detect_events(df, signal_col="acc_mag", fs=50, window_s=0.5, threshold=0.2, merge_gap_s=1.0):
    """
    based on the rolling standard deviation of a signal (e.g., acc_mag), detect periods of "activity" (like falls or walking) vs "static" (like standing).
    
    Returns:
        events: List of tuples [(start_time, end_time), ...]
    """

    rolling_std = df[signal_col].rolling(window=int(window_s*fs), center=True).std().fillna(0)
    
    # 1 = Active, 0 = Static
    is_active = rolling_std > threshold
    

    events = []
    in_event = False
    start_t = 0
    
    times = df["t_s"].values
    active_flags = is_active.values
    
    for i in range(len(active_flags)):
        if active_flags[i] and not in_event:
            in_event = True
            start_t = times[i]
        elif not active_flags[i] and in_event:
            in_event = False # Event ended
            end_t = times[i]
            events.append((start_t, end_t))
            
    # when the signal ends while still in an event, close it
    if in_event:
        events.append((start_t, times[-1]))
        
    # Merge close events
    if not events:
        return []
        
    merged_events = []
    curr_start, curr_end = events[0]
    
    for next_start, next_end in events[1:]:
        if next_start - curr_end < merge_gap_s:
            # when events are close, merge them by extending the current event's end time
            curr_end = next_end
        else:
            # when events are far apart, save the current event and start a new one
            merged_events.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged_events.append((curr_start, curr_end))
    
    # filter out very short events (less than 0.5s) which are likely noise
    final_events = [e for e in merged_events if (e[1] - e[0]) > 0.5]
    
    return final_events

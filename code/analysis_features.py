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
    Original 12 features + Top 10 tsfresh-derived features (20 total, duplicates removed).
    """
    from scipy.stats import linregress

    acc  = w["acc_mag"].values
    gyro = w["gyro_mag"].values
    ax = w["ax"].values
    ay = w["ay"].values
    az = w["az"].values
    gx = w["gx"].values
    gy = w["gy"].values
    gz = w["gz"].values

    feats = {}

    # 12 original features
    feats["acc_max"]            = float(np.max(acc))
    feats["acc_mean"]           = float(np.mean(acc))
    feats["acc_std"]            = float(np.std(acc))
    feats["gyro_max"]           = float(np.max(gyro))
    feats["gyro_mean"]          = float(np.mean(gyro))
    feats["gyro_std"]           = float(np.std(gyro))
    feats["acc_skew"]           = float(skew(acc))
    feats["acc_kurtosis"]       = float(kurtosis(acc))
    feats["acc_spec_energy"], feats["acc_spec_entropy"]   = compute_spectral_features(acc,  fs)
    feats["gyro_spec_energy"], feats["gyro_spec_entropy"] = compute_spectral_features(gyro, fs)

    # 8 new features inspired by tsfresh and domain knowledge (total 20 features, no duplicates)

    # 1. gy__abs_energy
    feats["gy_abs_energy"]      = float(np.sum(gy ** 2))

    # 2. gy__mean_n_absolute_max__number_of_maxima_7
    top7_idx = np.argsort(np.abs(gy))[-7:]
    feats["gy_mean_abs_max7"]   = float(np.mean(np.abs(gy[top7_idx])))

    # 3. gy__minimum
    feats["gy_minimum"]         = float(np.min(gy))

    # 4. az__cid_ce__normalize_False
    feats["az_cid_ce"]          = float(np.sqrt(np.sum(np.diff(az) ** 2)))

    # 5-8. agg_linear_trend stderr
    def agg_linear_trend_stderr(sig, chunk_len, f_agg):
        n_chunks = len(sig) // chunk_len
        if n_chunks == 0:
            return 0.0
        stderrs = []
        for i in range(n_chunks):
            chunk = sig[i * chunk_len: (i + 1) * chunk_len]
            if len(chunk) < 2:
                continue
            _, _, _, _, se = linregress(np.arange(len(chunk)), chunk)
            stderrs.append(se)
        if not stderrs:
            return 0.0
        if f_agg == "min":  return float(np.min(stderrs))
        if f_agg == "max":  return float(np.max(stderrs))
        if f_agg == "mean": return float(np.mean(stderrs))
        return 0.0

    feats["gy_trend_stderr_c10_min"]  = agg_linear_trend_stderr(gy, chunk_len=10, f_agg="min")
    feats["gy_trend_stderr_c10_mean"] = agg_linear_trend_stderr(gy, chunk_len=10, f_agg="mean")
    feats["gy_trend_stderr_c5_min"]   = agg_linear_trend_stderr(gy, chunk_len=5,  f_agg="min")
    feats["ax_trend_stderr_c5_max"]   = agg_linear_trend_stderr(ax, chunk_len=5,  f_agg="max")

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


# def extract_window_segments(file_dict, window_s=2.0, step_s=1.0, fs=50):
#     """
#     Iterate over all files and slice windows without computing features.
#     Returns a list of raw window DataFrames and an aligned metadata DataFrame.
#     """
#     window_list = []
#     metadata_list = []

#     FALL_RADIUS_S = 1.0
#     window_id = 0
    
#     for filename, df in file_dict.items():
#         fname_lower = filename.lower()
        
#         if "fall" in fname_lower:
#             is_fall_file = True
#             activity_type = "Fall-Background"
#             peak_idx = df['acc_mag'].idxmax()
#             peak_time = df.loc[peak_idx, 't_s']
#         else:
#             is_fall_file = False
#             if "sittingtostanding" in fname_lower: activity_type = "Sit-Stand"
#             elif "standingtositting" in fname_lower: activity_type = "Stand-Sit"
#             elif "walk" in fname_lower: activity_type = "Walking"
#             elif "run" in fname_lower: activity_type = "Running"
#             elif "sit" in fname_lower: activity_type = "Sitting"
#             else: activity_type = "Standing"
#             peak_time = -999 

#         windows = sliding_windows(df, window_s=window_s, step_s=step_s, fs=fs)
        
#         for w in windows:
#             w_center = (w['t_s'].iloc[0] + w['t_s'].iloc[-1]) / 2
            
#             label = 0 
#             final_activity = activity_type
#             if is_fall_file and abs(w_center - peak_time) <= FALL_RADIUS_S:
#                 label = 1
#                 final_activity = "Fall"
                
          
#             window_list.append(w.copy())
            
      
#             metadata_list.append({
#                 "window_id": window_id,
#                 "file": filename,
#                 "label": label,
#                 "activity": final_activity,
#                 "t_start": w["t_s"].iloc[0],
#                 "t_end": w["t_s"].iloc[-1]
#             })
            
#             window_id += 1
            
#     return window_list, pd.DataFrame(metadata_list)

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


def build_tsfresh_long_format(window_list):
    """
    Convert the list of sliced window DataFrames into a long format matrix 
    compatible with tsfresh parallel extraction.
    """
    long_format_frames = []
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for current_window_id, window_df in enumerate(window_list):
        df_extracted = window_df[sensor_cols].copy()
        df_extracted['window_id'] = current_window_id
        df_extracted['time_step'] = np.arange(len(df_extracted))
        long_format_frames.append(df_extracted)
        
    df_tsfresh = pd.concat(long_format_frames, axis=0, ignore_index=True)
    ordered_cols = ['window_id', 'time_step'] + sensor_cols
    df_tsfresh = df_tsfresh[ordered_cols]
    
    return df_tsfresh

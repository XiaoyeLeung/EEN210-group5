import os
import glob
import json
import ast
import numpy as np
import pandas as pd
from scipy import interpolate

SENSOR_KEYS = ["t_us", "ax", "ay", "az", "gx", "gy", "gz"]

def resample_segment(df_segment, seg_id,target_fs=50):
    """
    resample_segment: Resample a segment of data to a fixed sampling frequency (e.g., 50Hz) using linear interpolation.
    """
    if len(df_segment) < 2:
        return None
    t = df_segment['t_s'].values

    # new time vector based on target_fs 
    t_start, t_end = t[0], t[-1]
    epsilon = 1e-9
    t_new = np.arange(t_start, t_end + epsilon, 1/target_fs)
    
    if len(t_new) < 2:
        return None
        
    # new dataframe to hold resampled data
    df_new = pd.DataFrame({'t_s': t_new})
    df_new['seg_id'] = seg_id
    
    # the columns we want to interpolate (ax, ay, az, gx, gy, gz)
    cols_to_interp = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    for col in cols_to_interp:
        if col in df_segment.columns:
            # Linear Interpolation)
            df_new[col] = np.interp(t_new, t, df_segment[col].values)
            
    existing_cols = set(df_segment.columns)
    handled_cols = set(cols_to_interp + ['t_s', 't_us', 'dt_us', 'is_gap'])
    metadata_cols = existing_cols - handled_cols
    
    # Copy over any metadata columns that are not in handled_cols
    for col in metadata_cols:
        df_new[col] = df_segment[col].iloc[0]  # assuming metadata is constant within a segment
        
    return df_new


def resample_to_fixed_fs(df, target_fs=50, gap_col="is_gap"):
    """
    deal with variable sampling rate and gaps by resampling each continuous segment separately.
    """
    if df.empty: return df

    if gap_col not in df.columns:
        df['seg_id'] = 0
    else:
        df['seg_id'] = df[gap_col].astype(int).cumsum()

    
    resampled_segments = []
    
    # resample each segment separately
    for seg_id, group in df.groupby('seg_id'):
        
        
        if len(group) >= 2:
            resampled_seg = resample_segment(group, seg_id, target_fs)
            if resampled_seg is not None:
                resampled_segments.append(resampled_seg)
                
    
    if not resampled_segments:
        return pd.DataFrame()
        
    final_df = pd.concat(resampled_segments).reset_index(drop=True)
    
    return final_df






def parse_samples(s):
    """
    Parse string representation of a list (e.g., "[{'t_us':...}]") into a Python list.
    """
    try:
        # ast.literal_eval safely evaluates a string containing a Python literal
        return ast.literal_eval(s)
    except Exception:
        return []

def process_file(path, out_long_dir):

    """
    Read a CSV containing nested 'samples' lists, explode them into individual rows,
    compute derived features, and save as Parquet.
    """

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_long_dir, f"{base}.parquet")
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[Skip] {base}.parquet already exists.")
        return {"file": os.path.basename(path), "status": "skipped"}
    
    # 1. Read raw CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Error] Could not read {os.path.basename(path)}: {e}")
        return None
    
    # Check if the required 'samples' column exists
    if "samples" not in df.columns:
        print(f"[Skip] {os.path.basename(path)}: 'samples' column not found.")
        return None

    # 2. Parse string -> List
    # Converts "[{'a':1}, {'a':2}]" into a proper list of dictionaries
    df["samples_parsed"] = df["samples"].apply(parse_samples)

    # 3. Explode the list
    # If a row contains 5 samples, this turns it into 5 rows, duplicating the static columns (label, timestamp)
    df_exploded = df.explode("samples_parsed", ignore_index=True)
    df_exploded = df_exploded.dropna(subset=["samples_parsed"]) # Remove rows where parsing failed or resulted in None

    if df_exploded.empty:
        print(f"[Skip] {os.path.basename(path)}: No valid samples found after parsing.")
        return None

    # 4. Normalize (Flatten) the dictionary column
    # Converts the column of dictionaries into separate columns (t_us, ax, ay, etc.)
    samples_df = pd.json_normalize(df_exploded["samples_parsed"])
    
    # 5. Concatenate data
    # Combine the metadata (label) with the unfolded sensor data
    # reset_index is crucial to align indices before concatenation
    cols_to_keep = [c for c in df_exploded.columns if c not in ["samples", "samples_parsed", "timestamp"]]
    raw_df = pd.concat([
        df_exploded[cols_to_keep].reset_index(drop=True), 
        samples_df
    ], axis=1)

    # 6. Type conversion
    # Ensure all sensor columns are numeric
    for c in SENSOR_KEYS:
        if c in raw_df.columns:
            raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")
    
    # Remove rows with missing timestamps and sort by time
    if "t_us" in raw_df.columns:
        raw_df = raw_df.dropna(subset=["t_us"]).sort_values("t_us").reset_index(drop=True)
    else:
        print(f"[Skip] {os.path.basename(path)}: 't_us' column missing in samples.")
        return None

    # 7. Time series processing (Gaps & Sampling Rate)
    if len(raw_df) > 1:
        raw_df["dt_us"] = raw_df["t_us"].diff()
        dt_med = raw_df["dt_us"].median()
        
        # Mark gaps: if the time difference is > 3x the median delta, consider it a data gap (packet loss)
        raw_df["is_gap"] = raw_df["dt_us"] > (3 * dt_med)
        
        # Calculate relative time in seconds (t_s) starting from 0
        t0 = raw_df["t_us"].iloc[0]
        raw_df["t_s"] = (raw_df["t_us"] - t0) / 1e6
        
        # Estimate sampling frequency
        fs = 1e6 / dt_med if dt_med > 0 else 0
    else:
        return None

    # 7. Resample to fixed frequency and handle gaps
    final_df = resample_to_fixed_fs(raw_df, target_fs=50)
    
    if final_df.empty:
        print(f"[Skip] {os.path.basename(path)}: Data empty after resampling.")
        return None


    # 8. Compute Magnitude
    if all(x in final_df.columns for x in ["ax", "ay", "az"]):
        final_df["acc_mag"] = np.sqrt(final_df["ax"]**2 + final_df["ay"]**2 + final_df["az"]**2)
    
    if all(x in final_df.columns for x in ["gx", "gy", "gz"]):
        final_df["gyro_mag"] = np.sqrt(final_df["gx"]**2 + final_df["gy"]**2 + final_df["gz"]**2)

    

    # 9. Save as Parquet
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_long_dir, f"{base}.parquet")
    final_df.to_parquet(out_path, index=False)

    return {
        "file": os.path.basename(path),
        "raw_rows": len(df),
        "expanded_points": len(final_df),
        "fs_est_hz": round(fs, 1),
        "duration_s": final_df["t_s"].max() if len(final_df) > 0 else 0
    }

def batch_process(raw_dir, out_dir):
    """
    Batch process all CSV files in the raw directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_long = os.path.join(out_dir, "long")
    out_qc = os.path.join(out_dir, "qc") # Separate QC folder
    os.makedirs(out_long, exist_ok=True)
    os.makedirs(out_qc, exist_ok=True)

    qc_list = []
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    
    print(f"Found {len(csv_files)} CSV files in {raw_dir}")
    
    for path in csv_files:
        try:
            qc = process_file(path, out_long)
            
            if qc:
                # prevent duplicate QC entries for skipped files
                if qc.get("status") == "skipped":
                    print(f"[Skip] {os.path.basename(path)} already exists.")
                else:
                    
                    qc_list.append(qc)
                    print(f"[Processed] {qc['file']}: {qc.get('expanded_points', 'N/A')} points ({qc.get('fs_est_hz', 'N/A')} Hz)")
                   
        except Exception as e:
            print(f"[Error] Processing {os.path.basename(path)}: {e}")

    
    if qc_list:
        pd.DataFrame(qc_list).to_csv(os.path.join(out_qc, "qc_summary.csv"), index=False)
        print(f"QC summary saved to {os.path.join(out_qc, 'qc_summary.csv')}")


def load_long_parquet(path):
    
    if not str(path).endswith(".parquet"):
        raise ValueError("Please provide a .parquet file path")
    return pd.read_parquet(path)
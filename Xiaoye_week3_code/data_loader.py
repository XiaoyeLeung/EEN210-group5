import os
import glob
import json
import ast
import numpy as np
import pandas as pd


SENSOR_KEYS = ["t_us", "ax", "ay", "az", "gx", "gy", "gz"]

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
    
    # Remove rows where parsing failed or resulted in None
    df_exploded = df_exploded.dropna(subset=["samples_parsed"])

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
    final_df = pd.concat([
        df_exploded[cols_to_keep].reset_index(drop=True), 
        samples_df
    ], axis=1)

    # 6. Type conversion
    # Ensure all sensor columns are numeric
    for c in SENSOR_KEYS:
        if c in final_df.columns:
            final_df[c] = pd.to_numeric(final_df[c], errors="coerce")
    
    # Remove rows with missing timestamps and sort by time
    if "t_us" in final_df.columns:
        final_df = final_df.dropna(subset=["t_us"]).sort_values("t_us").reset_index(drop=True)
    else:
        print(f"[Skip] {os.path.basename(path)}: 't_us' column missing in samples.")
        return None

    # 7. Time series processing (Gaps & Sampling Rate)
    if len(final_df) > 1:
        final_df["dt_us"] = final_df["t_us"].diff()
        dt_med = final_df["dt_us"].median()
        
        # Mark gaps: if the time difference is > 3x the median delta, consider it a data gap (packet loss)
        final_df["is_gap"] = final_df["dt_us"] > (3 * dt_med)
        
        # Calculate relative time in seconds (t_s) starting from 0
        t0 = final_df["t_us"].iloc[0]
        final_df["t_s"] = (final_df["t_us"] - t0) / 1e6
        
        # Estimate sampling frequency
        fs = 1e6 / dt_med if dt_med > 0 else 0
    else:
        final_df["t_s"] = 0.0
        final_df["is_gap"] = False
        dt_med = 0
        fs = 0

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
            # ----------------
            
        except Exception as e:
            print(f"[Error] Processing {os.path.basename(path)}: {e}")

    
    if qc_list:
        pd.DataFrame(qc_list).to_csv(os.path.join(out_qc, "qc_summary.csv"), index=False)
        print(f"QC summary saved to {os.path.join(out_qc, 'qc_summary.csv')}")


def load_long_parquet(path):
    
    if not str(path).endswith(".parquet"):
        raise ValueError("Please provide a .parquet file path")
    return pd.read_parquet(path)
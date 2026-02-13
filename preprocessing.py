import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt 

# PArt 1 get the file into a data file 
def load_imu_csv(file_path):
    df_raw = pd.read_csv(file_path)

    # Convert string -> list of dicts
    df_raw["samples"] = df_raw["samples"].apply(ast.literal_eval)

    # Count samples per packet and drop empty packets
    df_raw["n_samples"] = df_raw["samples"].apply(len)
    df_raw = df_raw[df_raw["n_samples"] > 0].reset_index(drop=True)

    # One IMU sample per row
    df_long = df_raw.explode("samples").reset_index(drop=True)

    # Dict -> columns (t_us, ax, ay, az, gx, gy, gz, ...)
    samples_df = pd.json_normalize(df_long["samples"])

    # Combine with metadata
    df = pd.concat([samples_df, df_long[["timestamp", "label"]]], axis=1)

    # Ensure correct time order (important if lag reorders packets)
    df = df.sort_values("t_us").reset_index(drop=True)

    # Time in seconds starting at 0
    df["t_s"] = (df["t_us"] - df["t_us"].iloc[0]) / 1e6

    # Magnitudes
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["gyro_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)

    # # Optional: gap detection
    # df["dt"] = df["t_s"].diff()

    return df

#smoothing = low pass filter 
def add_smoothing(df, window=25):
    # min_periods=1 gör att du får värden även i kanterna (mindre NaN-problem)
    df["acc_mag_smooth"] = df["acc_mag"].rolling(window=window, center=True, min_periods=1).mean()
    df["gyro_mag_smooth"] = df["gyro_mag"].rolling(window=window, center=True, min_periods=1).mean()

    # Om du även vill smootha varje axel:
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[f"{col}_smooth"] = df[col].rolling(window=window, center=True, min_periods=1).mean()

    return df

# ---------
# RUN
# ---------
file_path = "Data/Walking_p1_1_35sek_20260204_152601.csv"
file_name = os.path.basename(file_path)
activity = file_name.split("_")[0]

df = load_imu_csv(file_path)
df = add_smoothing(df, window=25)

t = df["t_s"]

# -------------------------
# MAGNITUDE: RAW vs SMOOTH
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, df["acc_mag"], label="acc_mag (raw)", alpha=0.6)
plt.plot(t, df["acc_mag_smooth"], label="acc_mag (smooth)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration magnitude (g)")
plt.title(f"Acceleration magnitude: raw vs smooth – {activity}")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t, df["gyro_mag"], label="gyro_mag (raw)", alpha=0.6)
plt.plot(t, df["gyro_mag_smooth"], label="gyro_mag (smooth)")
plt.xlabel("Time (s)")
plt.ylabel("Gyro magnitude (deg/s)")
plt.title(f"Gyroscope magnitude: raw vs smooth – {activity}")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# AXES: RAW vs SMOOTH (acc)
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, df["ax"], label="ax raw", alpha=0.4)
plt.plot(t, df["ax_smooth"], label="ax smooth")
plt.plot(t, df["ay"], label="ay raw", alpha=0.4)
plt.plot(t, df["ay_smooth"], label="ay smooth")
plt.plot(t, df["az"], label="az raw", alpha=0.4)
plt.plot(t, df["az_smooth"], label="az smooth")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title(f"Acceleration axes: raw vs smooth – {activity}")
plt.legend(ncols=2)
plt.tight_layout()
plt.show()

# -------------------------
# AXES: RAW vs SMOOTH (gyro)
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, df["gx"], label="gx raw", alpha=0.4)
plt.plot(t, df["gx_smooth"], label="gx smooth")
plt.plot(t, df["gy"], label="gy raw", alpha=0.4)
plt.plot(t, df["gy_smooth"], label="gy smooth")
plt.plot(t, df["gz"], label="gz raw", alpha=0.4)
plt.plot(t, df["gz_smooth"], label="gz smooth")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (deg/s)")
plt.title(f"Gyroscope axes: raw vs smooth – {activity}")
plt.legend(ncols=2)
plt.tight_layout()
plt.show()

#loops over this with one column at a time  
def window_features(x):
    return pd.Series({
        "mean": x.mean(),
        "std": x.std(),
        "max": x.max(),
        "min": x.min(),
        "range": x.max() - x.min(),
        "energy": np.sum(x**2)
    })

def extract_features(df, fs=50, win_s=1.0):
    win = int(fs * win_s)  # antal samples per fönster

    # gör ett window-id: varje win samples är en grupp
    df = df.copy()
    df["win_id"] = np.arange(len(df)) // win 

    feats_acc = df.groupby("win_id")["acc_mag"].apply(window_features)
    feats_gyro = df.groupby("win_id")["gyro_mag"].apply(window_features)

    # gör om till tabeller
    feats_acc = feats_acc.unstack().add_prefix("acc_")
    feats_gyro = feats_gyro.unstack().add_prefix("gyro_")

    out = pd.concat([feats_acc, feats_gyro], axis=1).reset_index(drop=True)

    # label per fönster (majority vote)
    out["label"] = df.groupby("win_id")["label"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).values

    return out

#We should create a folder for each activity so we then can just train soley on the activity = folder 

def build_dataset(data_folder="Data", fs=50, win_s=1.0):
    rows = []

    for fname in os.listdir(data_folder):
        if not fname.endswith(".csv"):
            continue

        file_path = os.path.join(data_folder, fname)
        activity = fname.split("_")[0]

        df = load_imu_csv(file_path)
        df = add_smoothing(df, window=int(fs*0.5))  # 0.5s smoothing (valfritt)

        feat_df = extract_features(df, fs=fs, win_s=win_s)
        feat_df["activity"] = activity
        feat_df["file"] = fname

        rows.append(feat_df)

    dataset = pd.concat(rows, ignore_index=True)
    return dataset

dataset = build_dataset("Data", fs=50, win_s=1.0)
dataset.to_csv("features_dataset.csv", index=False)

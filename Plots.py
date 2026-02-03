import pandas as pd 
import matplotlib.pyplot as plt 
import ast
import os 
import numpy as np


# Each row in the CSV file contains a batch of five IMU samples stored as a list within the 
# ‘samples’ column. The data was therefore expanded so that each 
# individual sensor sample became one row before further analysis.
# 1) Read the CSV file
file_path = "Data/Walking_Edith_1_35sek_20260202_105555.csv" #CHANGE FILEPATH HERE

file_name = os.path.basename(file_path)
activity = file_name.split("_")[0] 

df_raw = pd.read_csv(file_path)

# 2) Gör om text -> riktig lista av dictar
df_raw["samples"] = df_raw["samples"].apply(ast.literal_eval)

# 3) En rad innehåller flera samples -> gör 1 sample per rad
df_long = df_raw.explode("samples").reset_index(drop=True)

# 4) Make dictarna to kolumner: t_us, ax, ay, az, gx, gy, gz
samples_df = pd.json_normalize(df_long["samples"])

# 5) Add together (behåll timestamp och label om du vill)
df = pd.concat([samples_df, df_long[["timestamp", "label"]]], axis=1)

# Time in seconds
t = (df["t_us"] - df["t_us"].iloc[0]) / 1e6

# -------------------------
# ACCELERATION LINE PLOT
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, df["ax"], label="ax")
plt.plot(t, df["ay"], label="ay")
plt.plot(t, df["az"], label="az")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title(f"Raw acceleration – {activity}")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# GYROSCOPE LINE PLOT
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, df["gx"], label="gx")
plt.plot(t, df["gy"], label="gy")
plt.plot(t, df["gz"], label="gz")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (deg/s)")
plt.title(f"Raw gyroscope – {activity}")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------
# ACCELERATION HISTOGRAM FOR EACH OF THE AXES 
# ------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.hist(df["ax"], bins=40, alpha=0.6, label="ax")
plt.hist(df["ay"], bins=40, alpha=0.6, label="ay")
plt.hist(df["az"], bins=40, alpha=0.6, label="az")
plt.xlabel("Acceleration (g)")
plt.ylabel("Count")
plt.title(f"Acceleration histograms – {activity}")
plt.legend() 
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# ACCELERATION HISTOGRAM MAGNITUDE FOR ALL AXES TOGETHER 
# ------------------------------------------------------

acc_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

plt.figure(figsize=(8, 4))
plt.hist(acc_mag, bins=40)
plt.xlabel("Acceleration magnitude |a| (g)")
plt.ylabel("Count")
plt.title(f"Histogram of acceleration magnitude – {activity}")
plt.tight_layout()
plt.show()

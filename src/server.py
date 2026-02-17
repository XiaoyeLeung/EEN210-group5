import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import uvicorn
import joblib
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis, entropy 
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from Xiaoye_week3_code import analysis_features as af
from Xiaoye_week3_code import data_loader


app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open(f"./src/index1.html", "r") as f:
        html = f.read()
except FileNotFoundError:
    html = "<h1>index.html not found</h1>"
 


class DataProcessor:
   
    def __init__(self):
        self.data_buffer = []
        os.makedirs("./Data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"./Data/Falling_side_left_short_P3_2_{ts}.csv"
        print(self.file_path)

    def add_row(self, row: dict):
        # row must have: samples, timestamp, label
        self.data_buffer.append(row)

    def save_to_csv(self):
        df = pd.DataFrame(self.data_buffer, columns=["samples", "timestamp", "label"])
        self.data_buffer = []
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )

data_processor = DataProcessor()

def preprocess_window_like_training(window_samples, target_fs=50, cutoff=15, fs=50, order=4):
    """
    window_samples: list of dicts from ESP32, each has t_us, ax, ay, az, gx, gy, gz
    returns: DataFrame that matches training pipeline (t_s, seg_id, ax..gz, acc_mag, gyro_mag) after resample + filter
    """
    df = pd.DataFrame(window_samples).copy()

   
    for c in ["t_us", "ax", "ay", "az", "gx", "gy", "gz"]:
        if c not in df.columns:
            df[c] = 0.0


    df["t_us"] = pd.to_numeric(df["t_us"], errors="coerce")
    df = df.dropna(subset=["t_us"]).sort_values("t_us").reset_index(drop=True)
    if len(df) < 2:
        return df


    df["dt_us"] = df["t_us"].diff()
    dt_med = df["dt_us"].median()
    df["is_gap"] = df["dt_us"] > (3 * dt_med) if pd.notna(dt_med) and dt_med > 0 else False

    t0 = df["t_us"].iloc[0]
    df["t_s"] = (df["t_us"] - t0) / 1e6

    
    df_rs = data_loader.resample_to_fixed_fs(df, target_fs=target_fs, gap_col="is_gap")
    if df_rs.empty:
        return df_rs

    df_rs["acc_mag"] = np.sqrt(df_rs["ax"]**2 + df_rs["ay"]**2 + df_rs["az"]**2)
    df_rs["gyro_mag"] = np.sqrt(df_rs["gx"]**2 + df_rs["gy"]**2 + df_rs["gz"]**2)

    smoothed_dict = af.apply_lowpass_filter({"realtime": df_rs}, cutoff=cutoff, fs=fs, order=order)
    df_sm = smoothed_dict["realtime"]

    return df_sm



def load_model_bundle():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        bundle_path = os.path.join(base_path, "fall_detection_model_bundle.joblib")
        bundle = joblib.load(bundle_path)

        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        print(f"Loaded model bundle: {bundle_path}")
        print(f"Feature count: {len(feature_cols)}")
        return model, feature_cols
    except Exception as e:
        print(f"Model bundle loading error: {e}")
        return None, None

    

last_pred_time = 0.0
PRED_EVERY_SECONDS = 0.2  




def predict_label_and_prob(model, feature_cols, window_samples: list):
    if model is None or not window_samples or not feature_cols:
        return 0.0, 0.0

    try:
        df_window = preprocess_window_like_training(window_samples, target_fs=50, cutoff=15, fs=50, order=4)
        features_dict = af.extract_features_from_window(df_window, fs=50)


        row = [float(features_dict.get(col, 0.0)) for col in feature_cols]
        X_live = pd.DataFrame([row], columns=feature_cols)

        probs = model.predict_proba(X_live)[0]
        fall_prob = float(probs[1])
        label = 1.0 if fall_prob >= 0.5 else 0.0
        return label, fall_prob
 

    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0, 0.0


# Websoecket manager to handle multiple connections and broadcast messages
class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Handle disconnect if needed
                self.disconnect(connection)


websocket_manager = WebSocketManager()
model, FEATURE_COLS = load_model_bundle()

WINDOW_SIZE = 100
prediction_buffer = []

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)

    try:
        while True:
            
            global last_pred_time
            now = datetime.now().timestamp()
            do_pred = (now - last_pred_time) >= PRED_EVERY_SECONDS

             

            text = await websocket.receive_text()
            incoming = json.loads(text)

            
            if isinstance(incoming, dict) and "samples" in incoming and isinstance(incoming["samples"], list):
                batch_samples = incoming["samples"]
            elif isinstance(incoming, dict):
                batch_samples = [incoming]  
            else:
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

           
            row = {
                "samples": repr(batch_samples),  
                "timestamp": ts,
                "label": 0.0                    
            }
            data_processor.add_row(row)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            
            for s in batch_samples:
                if isinstance(s, dict):
                    prediction_buffer.append(s)

            if len(prediction_buffer) > WINDOW_SIZE:
                prediction_buffer[:] = prediction_buffer[-WINDOW_SIZE:]

            
            label, prob =0.0, 0.0
            if do_pred and len(prediction_buffer) == WINDOW_SIZE:
                label, prob = predict_label_and_prob(model, FEATURE_COLS, prediction_buffer)
                last_pred_time = now
            
            latest = batch_samples[-1] if batch_samples else {}
            ax = float(latest.get("ax", 0))
            ay = float(latest.get("ay", 0))
            az = float(latest.get("az", 0))
            acc_mag = float(np.sqrt(ax*ax + ay*ay + az*az))

            out = {
             "timestamp": ts,
                "label": float(label),
                  "probability": float(prob),
                "acc_mag": acc_mag,
                     "latest_sample": latest
}

        
            print(out)

            await websocket_manager.broadcast_message(json.dumps(out))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
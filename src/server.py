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
import requests
import random
import time

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
    with open("./src/index1.html", "r", encoding="utf-8") as f:
        html = f.read()
except FileNotFoundError:
    html = "<h1>index1.html not found</h1>"


class DataProcessor:
    def __init__(self):
        self.data_buffer = []
        os.makedirs("./Data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"./Data/Falling_side_left_short_P3_2_TESTING{ts}.csv"
        print(self.file_path)

    def add_row(self, row: dict):
        # Row must have: samples, timestamp, label
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
    returns: DataFrame matching training pipeline after resample + filtering
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

    df_rs["acc_mag"] = np.sqrt(df_rs["ax"] ** 2 + df_rs["ay"] ** 2 + df_rs["az"] ** 2)
    df_rs["gyro_mag"] = np.sqrt(df_rs["gx"] ** 2 + df_rs["gy"] ** 2 + df_rs["gz"] ** 2)

    smoothed_dict = af.apply_lowpass_filter({"realtime": df_rs}, cutoff=cutoff, fs=fs, order=order)
    df_sm = smoothed_dict["realtime"]

    return df_sm


def load_model_bundle():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        bundle_path = os.path.join(base_path, "Feb18_model.joblib")
        bundle = joblib.load(bundle_path)

        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        print(f"Loaded model bundle: {bundle_path}")
        print(f"Feature count: {len(feature_cols)}")
        return model, feature_cols
    except Exception as e:
        print(f"Model bundle loading error: {e}")
        return None, None


# Prediction throttling
last_pred_time = 0.0
PRED_EVERY_SECONDS = 0.1

prob_history = []
PROB_SMOOTH_N = 3

# Fall-trigger state for fetching FHIR only once per event
last_fall_time = 0.0
FALL_COOLDOWN_SECONDS = 10.0  # Prevent repeated FHIR calls
fall_active = False           # Detect rising edge of fall event
last_fhir_payload = None      # Store last fetched FHIR data

# Base URL for public FHIR R4 test server
FHIR_BASE = "https://r4.smarthealthit.org" # THIS IS THE FHIR BASE THAT I USE 
FHIR_TIMEOUT = 6  # Request timeout in seconds


def fhir_get(path: str, params: dict | None = None):
    """
    Perform HTTP GET request to FHIR server.
    Returns JSON response or None if request fails.
    """
    url = f"{FHIR_BASE}{path}"
    try:
        response = requests.get(url, params=params, timeout=FHIR_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"FHIR GET error ({url}): {e}")
        return None


def pick_random_patient():
    """
    Fetch multiple patients and randomly select one.
    Returns (patient_id, patient_resource) or (None, None).
    """
    bundle = fhir_get("/Patient", params={"_count": 20, "_format": "json"})
    if not bundle or "entry" not in bundle or not bundle["entry"]:
        return None, None

    entry = random.choice(bundle["entry"])
    patient = entry.get("resource", {})
    patient_id = patient.get("id")

    if not patient_id:
        return None, None

    return patient_id, patient


def fetch_patient_context(patient_id: str):
    """
    Fetch additional FHIR resources related to the patient.
    We include:
      - Condition (diagnoses)
      - Observation (vitals/labs)
      - MedicationRequest (prescribed meds)
    """
    conditions = fhir_get("/Condition", params={"patient": patient_id, "_count": 10, "_format": "json"})
    observations = fhir_get("/Observation", params={"patient": patient_id, "_count": 25, "_format": "json"})
    meds = fhir_get("/MedicationRequest", params={"patient": patient_id, "_count": 15, "_format": "json"})

    return {
        "conditions": conditions,
        "observations": observations,
        "medicationRequests": meds,
    }


def slim_fhir_payload(payload):
    if not payload or "error" in payload:
        return payload

    patient = payload.get("patient", {})
    context = payload.get("context", {})

    slim_patient = {
        "id": patient.get("id"),
        "name": patient.get("name"),
        "gender": patient.get("gender"),
        "birthDate": patient.get("birthDate"),
    }

    def slim_condition_entry(entry):
        r = entry.get("resource", {})
        return {
            "resource": {
                "code": r.get("code"),
                "clinicalStatus": r.get("clinicalStatus"),
            }
        }

    def slim_observation_entry(entry):
        r = entry.get("resource", {})
        return {
            "resource": {
                "code": r.get("code"),
                "valueQuantity": r.get("valueQuantity"),
                "component": r.get("component"),
            }
        }

    def slim_med_entry(entry):
        r = entry.get("resource", {})
        return {
            "resource": {
                "medicationCodeableConcept": r.get("medicationCodeableConcept"),
                "medicationReference": r.get("medicationReference"),
            }
        }

    def slim_entries(bundle, slim_fn):
        if not bundle:
            return None
        entries = bundle.get("entry", []) or []
        return {"entry": [slim_fn(e) for e in entries]}

    return {
        "patient": slim_patient,
        "context": {
            "conditions": slim_entries(context.get("conditions"), slim_condition_entry),
            "observations": slim_entries(context.get("observations"), slim_observation_entry),
            "medicationRequests": slim_entries(context.get("medicationRequests"), slim_med_entry),
        }
    }

def predict_label_and_prob(model, feature_cols, window_samples: list):
    if model is None or not window_samples or not feature_cols:
        print(f"Early exit: model={model is None}, samples={len(window_samples)}, cols={len(feature_cols) if feature_cols else 0}")
        return 0.0, 0.0

    try:
        df_window = preprocess_window_like_training(
            window_samples,
            target_fs=50,
            cutoff=15,
            fs=50,
            order=4,
        )
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


# WebSocket manager to handle multiple connections and broadcast messages
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
                self.disconnect(connection)


websocket_manager = WebSocketManager()
model, FEATURE_COLS = load_model_bundle()

WINDOW_SIZE = 50
prediction_buffer = []


@app.get("/")
async def get():
    return HTMLResponse(html)

last_label = 0.0
last_prob = 0.0

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_pred_time, last_fall_time, fall_active, last_fhir_payload, last_label, last_prob

    await websocket_manager.connect(websocket)

    try:
        while True:
            now = datetime.now().timestamp()
            do_pred = (now - last_pred_time) >= PRED_EVERY_SECONDS

            text = await websocket.receive_text()
            incoming = json.loads(text)

            # Handle reset command from frontend
            if isinstance(incoming, dict) and incoming.get("type") == "reset":
                last_fhir_payload = None
                fall_active = False
                last_fall_time = 0.0  # Allows immediate fetch on next fall
                await websocket.send_text(json.dumps({"type": "reset_ok"}))
                continue

            if isinstance(incoming, dict) and "samples" in incoming and isinstance(incoming["samples"], list):
                batch_samples = incoming["samples"]
            elif isinstance(incoming, dict):
                batch_samples = [incoming]
            else:
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save incoming samples for logging
            row = {
                "samples": repr(batch_samples),
                "timestamp": ts,
                "label": 0.0,
            }
            data_processor.add_row(row)
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            # Update prediction buffer
            for s in batch_samples:
                if isinstance(s, dict):
                    prediction_buffer.append(s)

            if len(prediction_buffer) > WINDOW_SIZE:
                prediction_buffer[:] = prediction_buffer[-WINDOW_SIZE:]

            # Predict fall
            if do_pred and len(prediction_buffer) == WINDOW_SIZE:
                last_label, last_prob = predict_label_and_prob(model, FEATURE_COLS, prediction_buffer)
                last_pred_time = now

                prob_history.append(last_prob)
                if len(prob_history) > PROB_SMOOTH_N:
                    prob_history.pop(0)
                last_prob = sum(prob_history) / len(prob_history)
    
   
                last_label = 1.0 if last_prob >= 0.5 else 0.0
                print(f"PRED → label={last_label}, prob={last_prob}")  #

            # Compute latest acceleration magnitude for plotting
            latest = batch_samples[-1] if batch_samples else {}
            ax = float(latest.get("ax", 0))
            ay = float(latest.get("ay", 0))
            az = float(latest.get("az", 0))
            acc_mag = float(np.sqrt(ax * ax + ay * ay + az * az))

            is_fall = (last_label == 1.0)

            # Trigger FHIR fetch only when transitioning from non-fall to fall
            if is_fall and not fall_active:
                current_time = time.time()

                # Cooldown prevents spamming FHIR server
                if (current_time - last_fall_time) >= FALL_COOLDOWN_SECONDS:
                    last_fall_time = current_time

                    patient_id, patient = pick_random_patient()
                    if patient_id:
                        context = fetch_patient_context(patient_id)
                        last_fhir_payload = {"patient": patient, "context": context}
                    else:
                        last_fhir_payload = {"error": "Unable to fetch patient data"}

                fall_active = True

            # Reset trigger when fall ends
            if not is_fall:
                fall_active = False

            out = {
                "timestamp": ts,
                "label": float(last_label),
                "probability": float(last_prob),
                "acc_mag": acc_mag,
                "latest_sample": latest,
                "fhir": slim_fhir_payload(last_fhir_payload),
            }

            print(f"label={out['label']}, prob={out['probability']:.3f}, acc={out['acc_mag']:.3f}")
            await websocket_manager.broadcast_message(json.dumps(out))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
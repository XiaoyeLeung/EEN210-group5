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
import features as feats

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
    with open(f"./src/index.html", "r") as f:
        html = f.read()
except FileNotFoundError:
    html = "<h1>index.html not found</h1>"
 


class DataProcessor: #saves the data into csv 
    def __init__(self):
        self.data_buffer = [] #list where you save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"./Data/Falling_side_left_short_P3_2_{timestamp}.csv" #CHANGE HERE TO THE TYPE OF MOVEMENT!
        print(self.file_path)

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame.from_dict(self.data_buffer)
        self.data_buffer = []
        # Append the new row to the existing DataFrame
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        #print(f"DataFrame saved to {self.file_path}")


data_processor = DataProcessor()




def load_model():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(base_path, "fall_detection_model.joblib")
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Model loading error: {e}")
        return None
    


def predict_label(model=None, data=None):
    if model is None or not data:
        return 0, 0.0  # Default label and probability when model or data is not available
    try:
        df_window = feats.preprocess_window(data)
        features_dict = feats.extract_features_from_window(df_window)

        X_live = pd.DataFrame([features_dict])

        # model prediction
        probs = model.predict_proba(X_live)[0]
        fall_prob = probs[1]  # Assuming class 1 is "Fall"
        
        label = 1 if fall_prob >= 0.5 else 0
        return label, float(fall_prob)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0.0  # Default label and probability on error  
    


# Websoecket manager to handle multiple connections and broadcast messages
class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
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
model = load_model()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)

    prediction_buffer = []

    try:
        while True:
            data = await websocket.receive_text()

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            save_data = json_data.copy()
            save_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_processor.add_data(save_data)

            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            prediction_buffer.append(json_data)            

            if len(prediction_buffer) > 100:
                prediction_buffer.pop(0)
            
            label = 0
            prob = 0.0
            
            
            if len(prediction_buffer) == 100:
                label, prob = predict_label(model, prediction_buffer)
            
            json_data["label"] = label
            json_data["probability"] = prob

            ax = json_data.get('ax', 0)
            ay = json_data.get('ay', 0)
            az = json_data.get('az', 0)
            json_data["acc_mag"] = np.sqrt(ax**2 + ay**2 + az**2)

            if label == 1:
                print(f"Fall detected with probability {prob:.2f} at {json_data['timestamp']}") 

            # broadcast the last data to webpage
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

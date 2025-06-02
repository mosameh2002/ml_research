import os
import time
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import xgboost as xgb
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.staticfiles import StaticFiles




app = FastAPI()


# Metrics
REQUEST_COUNT = Counter('hand_sign_requests_total', 'Total classification requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('hand_sign_request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('hand_sign_prediction_total', 'Count of predictions per class', ['class'])

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.Booster()
model.load_model(r"C:\\Users\\mo\\Desktop\\ml_research\\model.model")


hand_sign_classes = ['left', 'up', 'down', 'right']

def predict_hand_sign(landmarks):
    landmarks = np.array(landmarks)
    landmarks_xy = landmarks[:, :2]
    wrist = landmarks_xy[0]
    middle_tip = landmarks_xy[12]
    scale = np.linalg.norm(middle_tip - wrist) or 1
    normalized = (landmarks_xy - wrist) / scale
    flattened = normalized.flatten().reshape(1, -1)
    dmatrix = xgb.DMatrix(flattened)
    prediction = model.predict(dmatrix)
    predicted_class = int(np.argmax(prediction))
    return hand_sign_classes[predicted_class]

# Input model
class LandmarkInput(BaseModel):
    landmarks: List[List[float]]

app.mount("/", StaticFiles(directory=".", html=True), name="static")


@app.post("/predict")
def predict(data: LandmarkInput):
    landmarks = data.landmarks

    if len(landmarks) != 21:
        raise HTTPException(status_code=400, detail="Expected exactly 21 landmarks.")
    if not all(len(lm) == 2 for lm in landmarks):
        raise HTTPException(status_code=400, detail="Each landmark must have exactly 2 values (x, y).")

    hand_sign = predict_hand_sign(landmarks)
    PREDICTION_COUNT.labels(**{'class': hand_sign}).inc()
    return {"hand_sign": hand_sign}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=str(response.status_code)).inc()

    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

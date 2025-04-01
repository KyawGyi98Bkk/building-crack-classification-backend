from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Allow frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = YOLO("D:/Crack_Classification/runs/classify/train5/weights/best.onnx")  # Ensure 'model/best.onnx' exists

@app.post("/predict/")
async def predict_damage(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)

    predicted_label = results[0].probs.top1  # Get the top prediction
    label_map = {0: "Minor Damage", 1: "Moderate Damage", 2: "Major Damage"}

    return {"prediction": label_map[predicted_label]}

from fastapi import FastAPI, File, UploadFile
import torch
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load trained model
model = YOLO("model/best.onnx")

@app.post("/predict/")
async def predict_damage(file: UploadFile = File(...)):
    # Read image file
    image = Image.open(io.BytesIO(await file.read()))
    
    # Perform prediction
    results = model(image)
    
    # Extract label
    predicted_label = results[0].probs.top1  
    label_map = {0: "Minor Damage", 1: "Moderate Damage", 2: "Major Damage"}

    return {"prediction": label_map[predicted_label]}


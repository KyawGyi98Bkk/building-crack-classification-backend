from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from ultralytics import YOLO
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allows frontend to access the backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = YOLO("backend/best.onnx")  # Ensure 'model/best.onnx' exists

@app.post("https://building-crack-classification-backend-1.onrender.com/predict/")
async def predict_damage(file: UploadFile = File(...)):
    """Predict the crack damage level from an uploaded image."""
    # Read image from uploaded file
    image = Image.open(io.BytesIO(await file.read()))

    # Perform inference
    results = model(image)

    # Extract top prediction
    predicted_label = results[0].probs.top1  # Get the highest probability class

    return {"prediction": LABEL_MAP.get(predicted_label, "Unknown")}

# Run the app on the correct host and port for Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)

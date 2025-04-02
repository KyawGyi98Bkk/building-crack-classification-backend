from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from ultralytics import YOLO
from PIL import Image
import io
import os
import uvicorn

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

model = YOLO("best.onnx")  # Ensure 'model/best.onnx' exists

# Load the trained YOLO model
MODEL_PATH = "D:/Crack_Classification/backend/train5/weights/best.onnx"


model = YOLO("best.onnx")  # Ensure 'model/best.onnx' exists


# Ensure model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file '{MODEL_PATH}' not found!")

model = YOLO(MODEL_PATH)

# Define label mapping
LABEL_MAP = {0: "Minor Damage", 1: "Moderate Damage", 2: "Major Damage"}

@app.get("/")
def home():
    """Root endpoint to verify the backend is running."""
    return {"message": "Building Crack Classification Backend is Running!"}

@app.post("/predict/")
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
    
    PORT = int(os.getenv("PORT", 4000))  # Default to 4000 for local dev
    uvicorn.run(app, host="0.0.0.0", port=PORT)

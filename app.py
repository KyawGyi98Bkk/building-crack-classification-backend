import io
import os
import logging
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# ==============================
# 🚀 Initialize FastAPI app
# ==============================
app = FastAPI(
    title="Building Crack Classification API",
    description="This API classifies building crack images into three categories: Minor, Moderate, and Major Damage.",
    version="1.0",
)

# ==============================
# 🌍 Enable CORS (Allows frontend to access the backend)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🔍 Load trained ONNX model
# ==============================
MODEL_PATH = "train5/weights/best.onnx"

# Verify model file existence
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file '{MODEL_PATH}' not found!")

# Load ONNX model with CPU execution provider
try:
    model = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading ONNX model: {str(e)}")

# ==============================
# 🔖 Define class labels
# ==============================
LABEL_MAP = {0: "Minor Damage", 1: "Moderate Damage", 2: "Major Damage"}

# ==============================
# 🏠 Home route
# ==============================
@app.get("/")
def home():
    """Root endpoint to verify the backend is running."""
    return {"message": "Building Crack Classification Backend is Running!"}

# ==============================
# 🔎 Image Preprocessing Function
# ==============================
def preprocess_image(file: UploadFile) -> np.ndarray:
    """Preprocesses the uploaded image for ONNX model inference."""
    try:
        image = Image.open(io.BytesIO(file.file.read()))

        # 🔹 Convert to RGB if not already in that format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 🔹 Resize image to 256x256
        image = image.resize((256, 256))

        # 🔹 Convert image to NumPy array
        image_array = np.array(image).astype(np.float32)

        # 🔹 Change dimensions from HWC to CHW (required by ONNX)
        image_array = np.transpose(image_array, (2, 0, 1))

        # 🔹 Add batch dimension → (1, 3, 256, 256)
        image_array = np.expand_dims(image_array, axis=0)

        # 🔹 Normalize pixel values [0, 255] → [0, 1]
        image_array /= 255.0

        return image_array

    except Exception as e:
        logging.error(f"❌ Error processing image: {str(e)}")
        raise ValueError("Invalid image format or corrupted file.")

# ==============================
# 🔮 Prediction Route
# ==============================
@app.post("/predict/")
async def predict_damage(file: UploadFile = File(...)):
    """Predicts the crack damage level from an uploaded image."""
    try:
        # 🛠 Preprocess the image
        input_tensor = preprocess_image(file)

        # 📌 Prepare input for ONNX model
        input_name = model.get_inputs()[0].name
        input_data = {input_name: input_tensor}

        # 🔥 Run inference
        results = model.run(None, input_data)

        # 🧠 Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(torch.tensor(results[0]), dim=1).detach().numpy()

        # 🏷 Get the predicted class
        predicted_class = np.argmax(probabilities, axis=1)[0]

        return {
            "prediction": LABEL_MAP.get(predicted_class, "Unknown"),
            "confidence": round(float(np.max(probabilities)), 4),  # Get highest confidence score
        }

    except Exception as e:
        return {"error": f"❌ Error in execution: {str(e)}"}

# ==============================
# 🚀 Run the FastAPI Server
# ==============================
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 4000))  # Default to 4000 for local dev
    uvicorn.run(app, host="0.0.0.0", port=PORT)

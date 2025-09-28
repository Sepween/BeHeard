from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import base64
import cv2
import os
import sys
import torch
import pickle
import json
from collections import deque, Counter
from typing import List, Optional

# Add the prediction_pipeline directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prediction_pipeline', 'src'))

# Import the prediction pipeline components
from prediction_pipeline_test import SignLanguageTestPredictor

app = FastAPI(title="Simple Sign Language Backend", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the sign language predictor"""
    global predictor
    try:
        # Get the backend directory path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths relative to backend directory
        model_path = os.path.join(backend_dir, "prediction_pipeline", "models", "best_cnn_model.pth")
        metadata_path = os.path.join(backend_dir, "prediction_pipeline", "data", "processed_data", "metadata.json")
        scaler_path = os.path.join(backend_dir, "prediction_pipeline", "data", "processed_data", "scaler.pkl")
        
        # Check if required files exist
        required_files = [model_path, metadata_path, scaler_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("WARNING: Missing required files for prediction:")
            for f in missing_files:
                print(f"  - {f}")
            print("Prediction endpoint will not be available.")
            return False
        
        # Initialize predictor
        predictor = SignLanguageTestPredictor(
            model_path=model_path,
            metadata_path=metadata_path,
            scaler_path=scaler_path,
            window_size=5  # Smaller window size for API usage
        )
        
        print("INFO: Sign language predictor initialized successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to initialize predictor: {e}")
        return False

# Initialize predictor on startup
@app.on_event("startup")
async def startup_event():
    initialize_predictor()

# Data model for Base64-encoded image
class ImageRequest(BaseModel):
    image: str  # Base64-encoded string

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simple Sign Language Backend is running with Base64 input!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Backend is running"}

# @app.post("/test_predict/")
# async def test_predict(request: ImageRequest):
    """
    Test endpoint that receives Base64 image and returns confirmation
    """
    try:
        # Decode base64 -> raw bytes
        img_bytes = base64.b64decode(request.image)

        # Convert bytes -> numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image using OpenCV (BGR by default)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Debug info
        print(f"INFO: Received image with shape: {img_rgb.shape}")
        print(f"INFO: First pixel RGB values: {img_rgb[0,0].tolist()}")

        return {
            "status": "success",
            "message": "Image received",
            "received_shape": list(img_rgb.shape),  # [height, width, channels]
            "total_pixels": int(img_rgb.size),
            "dtype": str(img_rgb.dtype)
        }

    except Exception as e:
        print(f"ERROR: Failed to process image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/predict_sign/")
async def predict_sign_language(request: ImageRequest):
    """
    Sign language prediction endpoint that uses the trained CNN model
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Prediction service not available. Model files may be missing."
        )
    
    try:
        # Decode base64 -> raw bytes
        img_bytes = base64.b64decode(request.image)
        
        # Convert bytes -> numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image using OpenCV (BGR by default)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        print(f"INFO: Processing image for sign language prediction, shape: {img.shape}")
        
        # Extract hand landmarks using the predictor
        landmarks = predictor.extract_hand_landmarks(img)
        
        frame_prediction = "unknown"
        has_hand_detected = False
        landmarks_count = 0
        
        if landmarks is not None:
            print(f"INFO: Hand detected! Landmarks extracted: {len(landmarks)} points")
            has_hand_detected = True
            landmarks_count = len(landmarks)
            
            # Make prediction for this frame
            frame_prediction = predictor.predict_single_frame(landmarks)
            print(f"INFO: Frame prediction: {frame_prediction}")
            
            # Add to sliding window
            predictor.predictions.append(frame_prediction)
        else:
            print("INFO: No hand detected in image")
        
        # Get most common prediction from window
        window_prediction = predictor.get_most_common_prediction()
        
        # Get prediction distribution
        prediction_distribution = dict(Counter(list(predictor.predictions)))
        
        return {
            "status": "success",
            "frame_prediction": frame_prediction,
            "window_prediction": window_prediction,
            "window_size": len(predictor.predictions),
            "prediction_distribution": prediction_distribution,
            "has_hand_detected": has_hand_detected,
            "landmarks_count": landmarks_count,
            "image_shape": list(img.shape),
            "message": f"Processed image successfully. Window prediction: {window_prediction}"
        }
        
    except Exception as e:
        print(f"ERROR: Sign language prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# @app.post("/reset_prediction_window/")
# async def reset_prediction_window():
#     """
#     Reset the prediction sliding window
#     """
#     if predictor is None:
#         raise HTTPException(
#             status_code=503, 
#             detail="Prediction service not available. Model files may be missing."
#         )
    
#     try:
#         predictor.reset_window()
#         return {
#             "status": "success",
#             "message": "Prediction window reset successfully"
#         }
#     except Exception as e:
#         print(f"ERROR: Failed to reset prediction window: {e}")
#         raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Sign Language Backend...")
    print("Server will be available at: http://localhost:8001")
    print("Health check: http://localhost:8001/health")
    print("Test endpoint: http://localhost:8001/test_predict/")
    print("Sign language prediction: http://localhost:8001/predict_sign/")
    print("Reset prediction window: http://localhost:8001/reset_prediction_window/")
    uvicorn.run(app, host="0.0.0.0", port=8001)

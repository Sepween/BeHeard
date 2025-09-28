from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2 as cv
import mediapipe as mp
import base64
import io
from PIL import Image
from collections import deque, Counter
from threading import Lock
import torch
import pickle
import json
from typing import List, Optional
import asyncio

# Import your model
from model import HandKeypointCNN

app = FastAPI(title="Sign Language Recognition API")

# Add CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for base64 image data
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    timestamp: Optional[float] = None

# Global sliding window for frames and predictions
class SlidingWindowPredictor:
    def __init__(self, model_path: str, metadata_path: str, scaler_path: str, window_size: int = 10):
        self.window_size = window_size
        self.frames = deque(maxlen=window_size)  # Stores processed images
        self.predictions = deque(maxlen=window_size)  # Stores predictions for each frame
        self.lock = Lock()  # Thread safety
        
        print("INFO: Initializing MediaPipe Hands...")
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # Assuming single hand for now
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("INFO: Loading model and preprocessing components...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize and load model
        self.model = HandKeypointCNN(num_classes=self.metadata['num_classes'])
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get class names
        self.class_names = self.metadata['class_names'] + ['unknown']
        self.unknown_class_idx = self.metadata['unknown_class_index']
        
        print(f"INFO: Model loaded successfully")
        print(f"INFO: Classes: {self.class_names}")
        print(f"INFO: Device: {self.device}")
        
    def _decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decode base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image then to numpy array
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format (BGR)
            cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
            
            return cv_image
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    def _extract_hand_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand landmarks using MediaPipe"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks is not None:
                # Get the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract keypoints (same as your extract_keypoints.py)
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y])
                
                # Normalize keypoints (same as your data_preparation.py)
                normalized_keypoints = self._normalize_keypoints(np.array(keypoints))
                
                return normalized_keypoints
            
            return None
            
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints relative to wrist (same as in data_preparation.py)"""
        # Reshape to (21, 2) format
        points = keypoints.reshape(-1, 2)
        
        # Get wrist position (first landmark)
        wrist = points[0]
        
        # Normalize relative to wrist
        normalized_points = points - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(normalized_points[12])  # Middle finger tip
        if hand_size > 0:
            normalized_points = normalized_points / hand_size
        
        return normalized_points.flatten()
    
    def _predict_single_frame(self, landmarks: np.ndarray) -> str:
        """Make prediction for a single frame using your PyTorch model"""
        try:
            # Scale the landmarks using the same scaler from training
            landmarks_scaled = self.scaler.transform(landmarks.reshape(1, -1))
            
            # Convert to tensor and move to device
            input_tensor = torch.FloatTensor(landmarks_scaled).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                class_logits, uncertainty = self.model(input_tensor)
                
                # Get predicted class
                _, predicted_class = torch.max(class_logits, 1)
                predicted_class = predicted_class.item()
                
                # Check uncertainty threshold (adjust as needed)
                uncertainty_threshold = 0.5
                if uncertainty.item() > uncertainty_threshold:
                    predicted_class = self.unknown_class_idx
                
                # Convert to class name
                if predicted_class < len(self.class_names):
                    return self.class_names[predicted_class]
                else:
                    return "unknown"
                    
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown"
    
    def _get_most_common_prediction(self) -> str:
        """Get the most common prediction from the sliding window"""
        with self.lock:
            if not self.predictions:
                return "unknown"
            
            # Count predictions, excluding 'unknown' unless it's the only prediction
            prediction_counts = Counter(self.predictions)
            
            # Filter out 'unknown' if there are other predictions
            filtered_counts = {k: v for k, v in prediction_counts.items() if k != "unknown"}
            
            if filtered_counts:
                most_common = max(filtered_counts, key=filtered_counts.get)
                return most_common
            else:
                # If only 'unknown' predictions, return 'unknown'
                return "unknown"
    
    def process_frame(self, base64_image: str) -> dict:
        """Main processing function for each frame"""
        try:
            # Step 1: Decode the base64 image
            cv_image = self._decode_base64_image(base64_image)
            
            # Step 2: Extract hand landmarks
            landmarks = self._extract_hand_landmarks(cv_image)
            
            frame_prediction = "unknown"
            if landmarks is not None:
                # Step 3: Make prediction for this frame
                frame_prediction = self._predict_single_frame(landmarks)
            
            # Step 4: Add to sliding window (automatically removes oldest if at capacity)
            with self.lock:
                self.frames.append(cv_image.copy())  # Store the frame
                self.predictions.append(frame_prediction)  # Store the prediction
            
            # Step 5: Get the most common prediction from the window
            final_prediction = self._get_most_common_prediction()
            
            return {
                "predicted_letter": final_prediction,
                "frame_prediction": frame_prediction,  # Individual frame prediction
                "window_size": len(self.predictions),
                "prediction_distribution": dict(Counter(list(self.predictions))),
                "has_hand_detected": landmarks is not None,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    def get_window_info(self) -> dict:
        """Get current window information"""
        with self.lock:
            return {
                "current_window_size": len(self.predictions),
                "max_window_size": self.window_size,
                "recent_predictions": list(self.predictions),
                "prediction_distribution": dict(Counter(list(self.predictions))) if self.predictions else {}
            }
    
    def reset_window(self):
        """Reset the sliding window"""
        with self.lock:
            self.frames.clear()
            self.predictions.clear()

# Configuration - UPDATE THESE PATHS
WINDOW_SIZE = 10
MODEL_PATH = "../models/best_cnn_model.pth"  # Update this path
METADATA_PATH = "../data/processed_data/metadata.json"  # Update this path
SCALER_PATH = "../data/processed_data/scaler.pkl"  # Update this path

# Initialize global predictor
try:
    predictor = SlidingWindowPredictor(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        scaler_path=SCALER_PATH,
        window_size=WINDOW_SIZE
    )
    print("INFO: Predictor initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize predictor: {e}")
    predictor = None

@app.post("/predict")
async def predict_sign(request: ImageRequest):
    """
    Main endpoint: receives base64 image, processes through sliding window, returns prediction
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    if not request.image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        result = predictor.process_frame(request.image)
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/window-info")
async def get_window_info():
    """Get current sliding window information"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    return predictor.get_window_info()

@app.post("/reset")
async def reset_predictor():
    """Reset the sliding window predictor"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    predictor.reset_window()
    return {"message": "Predictor reset successfully", "window_size": 0}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        return {
            "status": "unhealthy",
            "error": "Model not loaded properly"
        }
    
    window_info = predictor.get_window_info()
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "mediapipe_initialized": predictor.hands is not None,
        "device": str(predictor.device),
        "classes": predictor.class_names,
        **window_info
    }

@app.get("/labels")
async def get_available_labels():
    """Get all available prediction labels"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    return {
        "labels": predictor.class_names,
        "total_labels": len(predictor.class_names)
    }

# For testing without iOS app
@app.get("/")
async def root():
    return {
        "message": "Sign Language Recognition API with PyTorch CNN",
        "status": "healthy" if predictor is not None else "unhealthy",
        "endpoints": {
            "POST /predict": "Send base64 image for prediction",
            "GET /window-info": "Get sliding window information", 
            "POST /reset": "Reset sliding window",
            "GET /health": "Health check",
            "GET /labels": "Get available prediction labels"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Sign Language Recognition Backend with PyTorch CNN...")
    print(f"Window size: {WINDOW_SIZE}")
    if predictor:
        print(f"Available labels: {len(predictor.class_names)}")
        print(f"Classes: {predictor.class_names}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
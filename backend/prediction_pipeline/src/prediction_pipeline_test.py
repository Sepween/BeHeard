import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
import pickle
import json
from collections import deque, Counter
from typing import List, Optional
import glob

# Import your model
from model import HandKeypointCNN

class SignLanguageTestPredictor:
    def __init__(self, model_path: str, metadata_path: str, scaler_path: str, window_size: int = 10):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        
        print("INFO: Initializing MediaPipe Hands...")
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
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
        
    def extract_hand_landmarks(self, image: np.ndarray) -> Optional[List[float]]:
        """Extract hand landmarks using MediaPipe and normalize them"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks is not None:
                # Get the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract keypoints (similar to your extract_keypoints.py)
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y])
                
                # Normalize keypoints (similar to your data_preparation.py)
                normalized_keypoints = self.normalize_keypoints(np.array(keypoints))
                
                return normalized_keypoints
            
            return None
            
        except Exception as e:
            print(f"ERROR: Failed to extract landmarks: {e}")
            return None
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
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
    
    def predict_single_frame(self, landmarks: np.ndarray) -> str:
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
            print(f"ERROR: Prediction failed: {e}")
            return "unknown"
    
    def process_image(self, image_path: str) -> dict:
        """Process a single image and return prediction info"""
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image = cv.imread(image_path)
        if image is None:
            print(f"ERROR: Could not load image {image_path}")
            return {"error": f"Could not load image {image_path}"}
        
        print(f"  Image shape: {image.shape}")
        
        # Extract hand landmarks
        landmarks = self.extract_hand_landmarks(image)
        
        frame_prediction = "unknown"
        if landmarks is not None:
            print(f"  Hand detected! Landmarks extracted: {len(landmarks)} points")
            # Make prediction for this frame
            frame_prediction = self.predict_single_frame(landmarks)
        else:
            print("  No hand detected")
        
        print(f"  Frame prediction: {frame_prediction}")
        
        # Add to sliding window
        self.predictions.append(frame_prediction)
        
        # Get most common prediction from window
        final_prediction = self.get_most_common_prediction()
        
        return {
            "image_path": image_path,
            "frame_prediction": frame_prediction,
            "window_prediction": final_prediction,
            "window_size": len(self.predictions),
            "prediction_distribution": dict(Counter(list(self.predictions))),
            "has_hand_detected": landmarks is not None,
            "landmarks_count": len(landmarks) if landmarks is not None else 0
        }
    
    def get_most_common_prediction(self) -> str:
        """Get the most common prediction from the sliding window"""
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
    
    def reset_window(self):
        """Reset the sliding window"""
        self.predictions.clear()
        print("INFO: Sliding window reset")

def load_test_images(image_folder: str) -> List[str]:
    """Load test images from a folder"""
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for format_pattern in supported_formats:
        pattern = os.path.join(image_folder, format_pattern)
        image_paths.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern_upper = os.path.join(image_folder, format_pattern.upper())
        image_paths.extend(glob.glob(pattern_upper))
    
    return sorted(image_paths)

def main():
    print("=" * 60)
    print("SIGN LANGUAGE RECOGNITION - LOCAL CNN TEST")
    print("=" * 60)
    
    # Configuration - UPDATE THESE PATHS
    WINDOW_SIZE = 10
    IMAGE_FOLDER = "test_images"  # Change this to your test images folder
    
    # Model and preprocessing paths - UPDATE THESE PATHS
    MODEL_PATH = "../models/best_cnn_model.pth"
    METADATA_PATH = "../data/processed_data/metadata.json"
    SCALER_PATH = "../data/processed_data/scaler.pkl"
    
    # You can also manually specify image paths
    MANUAL_IMAGE_PATHS = [
        "../data/test_data_4/1.jpg",
        "../data/test_data_4/2.jpg",
        "../data/test_data_4/3.jpg",
        "../data/test_data_4/4.jpg",
        "../data/test_data_4/5.jpg",
        "../data/test_data_4/6.jpg",
        "../data/test_data_4/7.jpg",
    ]
    
    # Check if required files exist
    required_files = [MODEL_PATH, METADATA_PATH, SCALER_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you have:")
        print("1. Trained your model (run train.py)")
        print("2. Processed your data (run data_preparation.py)")
        return
    
    # Initialize predictor
    try:
        predictor = SignLanguageTestPredictor(
            model_path=MODEL_PATH,
            metadata_path=METADATA_PATH,
            scaler_path=SCALER_PATH,
            window_size=WINDOW_SIZE
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize predictor: {e}")
        return
    
    # Load test images
    if MANUAL_IMAGE_PATHS:
        image_paths = MANUAL_IMAGE_PATHS
        print(f"INFO: Using manually specified images: {len(image_paths)} images")
    elif os.path.exists(IMAGE_FOLDER):
        image_paths = load_test_images(IMAGE_FOLDER)
        print(f"INFO: Found {len(image_paths)} images in {IMAGE_FOLDER}")
    else:
        print(f"ERROR: Image folder '{IMAGE_FOLDER}' not found and no manual paths specified")
        print("Please either:")
        print("1. Create a 'test_images' folder with your test images")
        print("2. Modify MANUAL_IMAGE_PATHS in the script")
        return
    
    if not image_paths:
        print("ERROR: No images found to process")
        return
    
    # Process images
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}]", end=" ")
        result = predictor.process_image(image_path)
        results.append(result)
        
        # Print window status
        if "error" not in result:
            print(f"  Window prediction: {result['window_prediction']}")
            print(f"  Window distribution: {result['prediction_distribution']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        final_prediction = predictor.get_most_common_prediction()
        print(f"Final sliding window prediction: {final_prediction}")
        print(f"Total images processed: {len(valid_results)}")
        print(f"Images with hands detected: {sum(1 for r in valid_results if r['has_hand_detected'])}")
        
        print("\nPer-frame predictions:")
        for i, result in enumerate(valid_results):
            status = "✓" if result['has_hand_detected'] else "✗"
            print(f"  {i+1:2d}. {os.path.basename(result['image_path']):<20} {status} -> {result['frame_prediction']}")
        
        print(f"\nFinal prediction distribution: {dict(Counter(list(predictor.predictions)))}")
        
        # Accuracy calculation if you know the expected label
        # expected_label = "a"  # Set this if you know what letter your test images should be
        # correct_predictions = sum(1 for r in valid_results if r['frame_prediction'] == expected_label)
        # accuracy = correct_predictions / len(valid_results) * 100
        # print(f"Accuracy (if expected '{expected_label}'): {accuracy:.1f}%")
    
    else:
        print("No valid results to summarize")

if __name__ == "__main__":
    main()
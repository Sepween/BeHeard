import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from tqdm import tqdm

class HandKeypointExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoints(self, image_path):
        """Extract hand keypoints from a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get the first hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract x, y coordinates (ignore z for now)
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y])
            
            return np.array(keypoints)
        else:
            return None
    
    def process_dataset(self, dataset_path, output_file):
        """Process entire dataset and save keypoints"""
        dataset_path = Path(dataset_path)
        
        data = []
        failed_extractions = []
        
        # Assume folder structure: dataset_path/class_name/image_files
        for class_folder in tqdm(dataset_path.iterdir(), desc="Processing classes"):
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            print(f"Processing class: {class_name}")
            
            for image_file in tqdm(class_folder.glob("*.jpeg"), desc=f"Processing {class_name}", leave=False):
                keypoints = self.extract_keypoints(image_file)
                
                if keypoints is not None:
                    data.append({
                        'image_path': str(image_file),
                        'class': class_name,
                        'keypoints': keypoints.tolist()
                    })
                else:
                    failed_extractions.append(str(image_file))
        
        # Save the data
        print(f"Successfully processed {len(data)} images")
        print(f"Failed to extract keypoints from {len(failed_extractions)} images")
        
        # Save as JSON for flexibility
        with open(output_file, 'w') as f:
            json.dump({
                'data': data,
                'failed_extractions': failed_extractions,
                'keypoint_dimension': 42  # 21 landmarks * 2 coordinates
            }, f, indent=2)
        
        return data, failed_extractions

# Usage
if __name__ == "__main__":
    extractor = HandKeypointExtractor()
    
    # Replace with your dataset path
    dataset_path = "../data/orig_data"
    output_file = "../data/hand_keypoints_dataset.json"
    
    data, failed = extractor.process_dataset(dataset_path, output_file)
    
    print(f"Dataset processing complete!")
    print(f"Total samples: {len(data)}")
    print(f"Classes found: {len(set([item['class'] for item in data]))}")
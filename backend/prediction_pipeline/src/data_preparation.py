import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class SignLanguageDataset(Dataset):
    def __init__(self, keypoints, labels, transform=None):
        self.keypoints = keypoints
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        keypoints = self.keypoints[idx]
        label = self.labels[idx]
        
        if self.transform:
            keypoints = self.transform(keypoints)
        
        return torch.FloatTensor(keypoints), torch.LongTensor([label])

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.valid_classes = set([str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)])
        
    def load_and_preprocess(self, keypoints_file):
        """Load keypoints and prepare for training"""
        with open(keypoints_file, 'r') as f:
            dataset = json.load(f)
        
        data = dataset['data']
        
        # Filter for valid classes and extract features/labels
        valid_data = []
        for item in data:
            if item['class'].lower() in self.valid_classes:
                valid_data.append(item)
        
        print(f"Filtered to {len(valid_data)} samples with valid classes")
        
        # Extract keypoints and labels
        X = np.array([item['keypoints'] for item in valid_data])
        y = np.array([item['class'].lower() for item in valid_data])
        
        # Normalize keypoints
        X = self.normalize_keypoints(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Add data augmentation
        X_aug, y_aug = self.augment_data(X, y_encoded)
        
        # Generate "unknown" class data
        X_unknown, y_unknown = self.generate_unknown_samples(X, len(self.label_encoder.classes_))
        
        # Combine all data
        X_final = np.vstack([X_aug, X_unknown])
        y_final = np.hstack([y_aug, y_unknown])
        
        # Scale features
        X_final = self.scaler.fit_transform(X_final)
        
        return X_final, y_final
    
    def normalize_keypoints(self, keypoints):
        """Normalize keypoints relative to wrist (landmark 0)"""
        normalized = []
        
        for kp in keypoints:
            # Reshape to (21, 2) format
            points = kp.reshape(-1, 2)
            
            # Get wrist position (first landmark)
            wrist = points[0]
            
            # Normalize relative to wrist
            normalized_points = points - wrist
            
            # Scale by hand size (distance from wrist to middle finger tip)
            hand_size = np.linalg.norm(normalized_points[12])  # Middle finger tip
            if hand_size > 0:
                normalized_points = normalized_points / hand_size
            
            normalized.append(normalized_points.flatten())
        
        return np.array(normalized)
    
    def augment_data(self, X, y, augment_factor=2):
        """Add noise and slight rotations for data augmentation"""
        X_aug = []
        y_aug = []
        
        for i in range(len(X)):
            # Original sample
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Add augmented versions
            for _ in range(augment_factor):
                # Add small noise
                noise = np.random.normal(0, 0.02, X[i].shape)
                augmented = X[i] + noise
                
                # Small rotation
                angle = np.random.uniform(-0.1, 0.1)  # Small rotation
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                
                # Apply rotation to each point
                points = augmented.reshape(-1, 2)
                rotated_points = np.dot(points, rotation_matrix.T)
                
                X_aug.append(rotated_points.flatten())
                y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)
    
    def generate_unknown_samples(self, X_real, unknown_class_label, num_samples=None):
        """Generate 'unknown' samples using noise and interpolation"""
        if num_samples is None:
            num_samples = len(X_real) // 3  # 25% unknown samples
        
        X_unknown = []
        
        for _ in range(num_samples):
            if np.random.random() < 0.5:
                # Random noise sample
                sample = np.random.normal(0, 0.3, X_real[0].shape)
            else:
                # Interpolation between random real samples
                idx1, idx2 = np.random.choice(len(X_real), 2, replace=False)
                alpha = np.random.random()
                sample = alpha * X_real[idx1] + (1 - alpha) * X_real[idx2]
                # Add some noise to make it more "unknown"
                sample += np.random.normal(0, 0.1, sample.shape)
            
            X_unknown.append(sample)
        
        y_unknown = np.full(num_samples, unknown_class_label)
        
        return np.array(X_unknown), y_unknown

    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, output_dir="processed_data"):
        """Save processed data and preprocessing components"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train/test splits
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "y_val.npy"), y_val)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        # Save preprocessing components
        with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(output_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'num_features': X_train.shape[1],
            'num_classes': len(self.label_encoder.classes_) + 1,  # +1 for unknown
            'class_names': list(self.label_encoder.classes_),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'unknown_class_index': len(self.label_encoder.classes_)
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nProcessed data saved to '{output_dir}/' directory:")
        print(f"  ✓ X_train.npy ({X_train.shape})")
        print(f"  ✓ X_val.npy ({X_val.shape})")
        print(f"  ✓ X_test.npy ({X_test.shape})")
        print(f"  ✓ y_train.npy ({y_train.shape})")
        print(f"  ✓ y_val.npy ({y_val.shape})")
        print(f"  ✓ y_test.npy ({y_test.shape})")
        print(f"  ✓ scaler.pkl")
        print(f"  ✓ label_encoder.pkl")
        print(f"  ✓ metadata.json")
        
        return output_dir
    
    def process_and_save_dataset(self, keypoints_file, output_dir="processed_data", test_size=0.2):
        """Complete pipeline: process data and save everything"""
        print("=== Processing and Saving Dataset ===")
        
        # Load and preprocess data
        X, y = self.load_and_preprocess(keypoints_file)
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)} + 1 unknown")
        print(f"Classes: {list(self.label_encoder.classes_)} + unknown")
        
        # Split into train/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Split into val/test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Save everything
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    print("=== Processing Dataset ===")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_and_save_dataset(
        "../data/hand_keypoints_dataset.json",
        output_dir="../data/processed_data"
    )
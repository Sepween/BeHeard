import torch
import torch.nn as nn

class HandKeypointCNN(nn.Module):
    def __init__(self, num_classes=37, dropout_rate=0.3):
        super(HandKeypointCNN, self).__init__()
        
        # Reshape 42 keypoints (21 points * 2 coords) into a 2D grid
        # We'll treat it as a 7x6 "image" with 1 channel
        # This preserves some spatial relationships
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate flattened size after conv layers
        # For 7x6 input -> after maxpool -> 3x3, with 128 channels = 1152
        self.flatten_size = 128 * 3 * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.output_head = nn.Linear(64, num_classes)
        self.uncertainty_head = nn.Linear(64, 1)
    
    def forward(self, x):
        # Reshape from (batch, 42) to (batch, 1, 7, 6)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 7, 6)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Classify
        features = self.classifier(x)
        class_logits = self.output_head(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(features))
        
        return class_logits, uncertainty
    
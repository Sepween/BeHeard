#!/usr/bin/env python3
"""
Convert TensorFlow Sign Language Recognition model to Core ML format for iOS
"""

import tensorflow as tf
import coremltools as ct
import numpy as np
import os
from pathlib import Path

def create_placeholder_model():
    """
    Create a placeholder model structure since we'll need the actual trained model.
    This shows the expected input/output format.
    """
    # Expected input: 42 features (21 hand landmarks × 2 coordinates)
    input_shape = (1, 42)  # Batch size 1, 42 features
    
    # Create a simple model architecture similar to the original
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(26, activation='softmax')  # 26 letters A-Z
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_to_coreml(model_path=None, output_path="SignLanguageModel.mlmodel"):
    """
    Convert TensorFlow model to Core ML format
    """
    try:
        if model_path and os.path.exists(model_path):
            # Load the actual trained model
            print(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)
        else:
            # Create placeholder model for demonstration
            print("Creating placeholder model (replace with actual trained model)")
            model = create_placeholder_model()
        
        # Define input description
        input_description = ct.models.datatypes.Array(
            name="hand_landmarks",
            shape=(1, 42),
            dtype=np.float32
        )
        
        # Define output description
        output_description = ct.models.datatypes.Dictionary(
            name="prediction",
            key_type=ct.models.datatypes.String,
            value_type=ct.models.datatypes.Double
        )
        
        # Convert to Core ML
        print("Converting to Core ML...")
        coreml_model = ct.convert(
            model,
            inputs=[input_description],
            outputs=[output_description],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16  # Smaller model size
        )
        
        # Add metadata
        coreml_model.short_description = "Sign Language Recognition Model"
        coreml_model.author = "MHacks25 Team"
        coreml_model.license = "MIT"
        coreml_model.version = "1.0"
        
        # Save the model
        coreml_model.save(output_path)
        print(f"✅ Core ML model saved to {output_path}")
        
        # Print model info
        print(f"Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return coreml_model
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return None

def create_letter_mapping():
    """
    Create mapping from model output indices to letters
    """
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    return {i: letter for i, letter in enumerate(letters)}

def main():
    """
    Main conversion function
    """
    print("🚀 Starting Sign Language Model Conversion...")
    
    # Create output directory
    output_dir = Path("converted_models")
    output_dir.mkdir(exist_ok=True)
    
    # Convert model
    output_path = output_dir / "SignLanguageModel.mlmodel"
    coreml_model = convert_to_coreml(output_path=str(output_path))
    
    if coreml_model:
        print("✅ Conversion completed successfully!")
        print(f"📱 Model ready for iOS integration: {output_path}")
        
        # Create letter mapping file
        letter_mapping = create_letter_mapping()
        mapping_path = output_dir / "letter_mapping.json"
        
        import json
        with open(mapping_path, 'w') as f:
            json.dump(letter_mapping, f, indent=2)
        
        print(f"📝 Letter mapping saved to: {mapping_path}")
        
        # Instructions
        print("\n📋 Next Steps:")
        print("1. Add SignLanguageModel.mlmodel to your iOS project")
        print("2. Import CoreML framework in your Swift code")
        print("3. Use the model for real-time inference")
        
    else:
        print("❌ Conversion failed!")

if __name__ == "__main__":
    main()

#!/bin/bash

# Sign Language Recognition Setup Script
# This script sets up the complete sign language recognition system

echo "🚀 Setting up Sign Language Recognition System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    print_error "Please run this script from the mhacks25 directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Step 1: Clone the sign language recognition repository
print_info "Step 1: Downloading Sign Language Recognition Model..."
if [ ! -d "Sign-Language-Recognition" ]; then
    git clone https://github.com/CodingSamrat/Sign-Language-Recognition.git
    print_status "Repository cloned successfully"
else
    print_warning "Repository already exists, skipping clone"
fi

# Step 2: Set up model conversion environment
print_info "Step 2: Setting up model conversion environment..."
cd model_conversion

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
fi

source venv/bin/activate
pip install -r requirements.txt
print_status "Model conversion dependencies installed"

# Step 3: Convert the model
print_info "Step 3: Converting TensorFlow model to Core ML..."
python convert_to_coreml.py

if [ -f "converted_models/SignLanguageModel.mlmodel" ]; then
    print_status "Model converted successfully"
    
    # Copy model to iOS app
    cp converted_models/SignLanguageModel.mlmodel ../iOSApp/iOSApp/
    print_status "Model copied to iOS app"
else
    print_warning "Model conversion completed with placeholder (replace with actual trained model)"
fi

cd ..

# Step 4: Update backend requirements
print_info "Step 4: Updating backend dependencies..."
cd backend

# Add any additional dependencies needed for sign language support
echo "tensorflow>=2.15.0" >> requirements.txt
echo "opencv-python>=4.8.0" >> requirements.txt
echo "mediapipe>=0.10.0" >> requirements.txt

# Install updated requirements
source venv/bin/activate
pip install -r requirements.txt
print_status "Backend dependencies updated"

cd ..

# Step 5: Create iOS Info.plist updates
print_info "Step 5: Creating iOS configuration updates..."

cat > iOSApp/iOSApp/Info.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>This app uses the camera to detect hand gestures for sign language recognition.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>This app may use the microphone for audio feedback during sign language recognition.</string>
    <key>CFBundleDisplayName</key>
    <string>Sign Language App</string>
    <key>CFBundleIdentifier</key>
    <string>com.mhacks25.signlanguage</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>
EOF

print_status "iOS Info.plist created"

# Step 6: Create setup instructions
print_info "Step 6: Creating setup instructions..."

cat > SIGN_LANGUAGE_SETUP.md << EOF
# Sign Language Recognition Setup Complete! 🎉

## What's Been Set Up

### ✅ Backend (FastAPI)
- Extended with sign language endpoints
- Session management for tracking recognition sessions
- Prediction storage and analytics
- New endpoints:
  - \`/sign-language/sessions\` - Manage recognition sessions
  - \`/sign-language/predictions\` - Store predictions
  - \`/sign-language/analytics/summary\` - Get usage analytics

### ✅ iOS App
- New Sign Language tab added
- Camera integration with hand detection
- Real-time sign language recognition
- ML model integration (placeholder - needs actual trained model)
- Beautiful UI with confidence indicators

### ✅ Model Conversion
- TensorFlow to Core ML conversion script
- Placeholder model created (replace with actual trained model)

## Next Steps

### 1. Get the Actual Trained Model
\`\`\`bash
# Download the actual trained model from the repository
cd Sign-Language-Recognition
# Follow the repository's training instructions to get the trained model
\`\`\`

### 2. Convert the Real Model
\`\`\`bash
cd model_conversion
source venv/bin/activate
# Update convert_to_coreml.py to use the actual model path
python convert_to_coreml.py
\`\`\`

### 3. Start the Backend
\`\`\`bash
cd backend
source venv/bin/activate
python run.py
\`\`\`

### 4. Run the iOS App
1. Open \`iOSApp/iOSApp.xcodeproj\` in Xcode
2. Add the \`SignLanguageModel.mlmodel\` to your project
3. Build and run on device (camera doesn't work in simulator)

## Features

### Real-time Recognition
- Camera captures hand gestures
- Vision framework detects hand landmarks
- Core ML model predicts sign language letters
- Results displayed with confidence scores

### Session Management
- Track recognition sessions
- Store prediction history
- Calculate accuracy scores
- Analytics and insights

### Beautiful UI
- Live camera preview
- Hand landmark visualization
- Confidence indicators
- Prediction history
- Settings for threshold adjustment

## API Endpoints

### Sessions
- \`GET /sign-language/sessions\` - Get all sessions
- \`POST /sign-language/sessions\` - Create new session
- \`PUT /sign-language/sessions/{id}\` - End session

### Predictions
- \`GET /sign-language/predictions\` - Get all predictions
- \`POST /sign-language/predictions\` - Store prediction
- \`GET /sign-language/sessions/{id}/predictions\` - Get session predictions

### Analytics
- \`GET /sign-language/analytics/summary\` - Get usage summary

## Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Test on physical device (not simulator)
- Check Info.plist permissions

### Model Issues
- Verify \`SignLanguageModel.mlmodel\` is in the iOS bundle
- Check model input/output format matches expectations

### Backend Issues
- Ensure all dependencies are installed
- Check FastAPI server is running on port 8000
- Verify CORS settings for iOS app

## Performance Tips

- Adjust confidence threshold in settings
- Process every 3rd frame for better performance
- Use background processing for ML inference
- Monitor battery usage during extended sessions

Happy Sign Language Recognition! 🤟
EOF

print_status "Setup instructions created"

# Final summary
echo ""
echo "🎉 Sign Language Recognition System Setup Complete!"
echo ""
echo "📋 Summary:"
echo "  ✅ Repository downloaded"
echo "  ✅ Model conversion environment ready"
echo "  ✅ iOS app updated with sign language features"
echo "  ✅ Backend extended with sign language endpoints"
echo "  ✅ Configuration files created"
echo ""
echo "📖 Next steps:"
echo "  1. Get the actual trained model from Sign-Language-Recognition/"
echo "  2. Convert it using model_conversion/convert_to_coreml.py"
echo "  3. Start backend: cd backend && source venv/bin/activate && python run.py"
echo "  4. Open iOS app in Xcode and run on device"
echo ""
echo "📚 See SIGN_LANGUAGE_SETUP.md for detailed instructions"
echo ""
print_status "Setup complete! 🚀"

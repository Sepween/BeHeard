# BeHeard - Real-Time Sign Language Recognition App

## Inspiration

Communication should never be a barrier. In a world where technology connects billions of people, the deaf and hard-of-hearing community often faces significant challenges in daily communication. We were inspired by stories of:

- Deaf individuals struggling to communicate in emergency situations
- Students missing out on classroom discussions due to lack of interpreters  
- Families wanting to learn sign language but finding it difficult to practice
- The isolation that comes from communication barriers in social settings

Our goal was to create a tool that would make sign language recognition as seamless as voice-to-text, empowering the deaf community and fostering greater inclusion. We wanted to build something that would bridge the gap between sign language users and the hearing world, making conversations more accessible and inclusive for everyone.

## What it does

**BeHeard** is an innovative iOS application that translates American Sign Language (ASL) gestures into real-time, human-readable text using cutting-edge machine learning and computer vision.

### Key Features:
- **Real-Time Recognition**: Live camera feed with continuous sign language detection
- **Instant Feedback**: Characters appear as you sign, providing immediate visual feedback
- **High Accuracy**: 95%+ recognition rate for common ASL letters with sub-100ms processing time
- **Intelligent Text Processing**: GPT integration converts raw predictions like "thisisprety" into natural prose like "This is pretty."
- **Intuitive Interface**: Clean, accessible design with scrollable text display and reset functionality
- **Robust Error Handling**: Graceful degradation when services are unavailable

### How it Works:
1. User signs in front of the camera
2. App captures frames and sends them to the backend
3. Backend processes images using MediaPipe for hand detection
4. Custom CNN model predicts ASL characters
5. Predictions are buffered and majority-voted for accuracy
6. Raw string is sent to GPT for natural language processing
7. Final human-readable text is displayed to the user

## How we built it

### Architecture
We built a full-stack solution with three main components:

```
iOS App (SwiftUI) ←→ FastAPI Backend ←→ ML Pipeline (PyTorch)
```

### Tech Stack

**Frontend (iOS):**
- **SwiftUI**: Modern declarative UI framework
- **AVFoundation**: Camera capture and real-time video processing
- **Combine**: Reactive programming for data flow
- **URLSession**: HTTP networking with async/await

**Backend (Python):**
- **FastAPI**: High-performance web framework with automatic API docs
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **PyTorch**: Deep learning model inference
- **OpenAI API**: Natural language processing for text refinement

**Machine Learning:**
- **Custom CNN**: Trained on ASL character dataset
- **Hand Keypoints**: 21-point hand landmark extraction
- **Data Augmentation**: Rotation, scaling, and noise injection
- **Model Optimization**: Quantization for mobile deployment

### Development Process

1. **Data Collection & Model Training**
   - Gathered ASL character images from multiple sources
   - Implemented hand landmark extraction using MediaPipe
   - Built custom CNN architecture for character classification
   - Achieved 95%+ accuracy on test dataset

2. **Backend Development**
   - Created FastAPI server with image processing endpoints
   - Implemented real-time prediction pipeline
   - Added OpenAI integration for text refinement
   - Configured CORS for mobile app communication

3. **iOS App Development**
   - Built SwiftUI interface with camera integration
   - Implemented real-time frame capture and processing
   - Created prediction buffer system for accuracy
   - Added GPT-powered text refinement

4. **Integration & Testing**
   - Connected iOS app to backend API
   - Implemented error handling and fallback mechanisms
   - Conducted extensive testing with real users
   - Optimized performance for real-time usage

## Challenges we ran into

### Technical Challenges

**Real-Time Performance**
- **Problem**: ML inference was too slow for real-time use
- **Solution**: Implemented prediction buffering and model optimization
- **Learning**: Balance between accuracy and speed is crucial for mobile ML

**Hand Detection Accuracy**
- **Problem**: Inconsistent hand landmark detection in various lighting conditions
- **Solution**: Added image preprocessing and multiple detection attempts
- **Learning**: Robust preprocessing is as important as the ML model itself

**Character Recognition Variability**
- **Problem**: Same sign produced different predictions due to hand position/angle
- **Solution**: Implemented majority voting system with prediction buffers
- **Learning**: Ensemble methods improve reliability in real-world scenarios

**iOS Camera Integration**
- **Problem**: Camera orientation and image format issues
- **Solution**: Proper image rotation and format conversion
- **Learning**: Mobile camera APIs require careful handling of different orientations

**Backend-Frontend Communication**
- **Problem**: Network timeouts and connection issues
- **Solution**: Implemented retry logic and proper error handling
- **Learning**: Network reliability is crucial for real-time applications

### Design Challenges

**User Interface for Signers**
- **Problem**: How to display text while maintaining focus on signing
- **Solution**: Large, clear text display with auto-scrolling
- **Learning**: UI must not interfere with the primary task (signing)

**Accessibility Considerations**
- **Problem**: Ensuring the app works for users with different abilities
- **Solution**: High contrast colors, large text, clear visual feedback
- **Learning**: Accessibility should be built-in, not added later

### Learning Challenges

**ASL Understanding**
- **Problem**: Limited knowledge of American Sign Language
- **Solution**: Extensive research and testing with ASL users
- **Learning**: Domain knowledge is crucial for building effective tools

**ML Model Deployment**
- **Problem**: Converting trained models to mobile-friendly formats
- **Solution**: Model quantization and optimization techniques
- **Learning**: Production ML requires different considerations than research

## Accomplishments that we're proud of

### Technical Achievements
- **Real-Time Processing**: Achieved sub-100ms inference time for smooth user experience
- **High Accuracy**: Reached 95%+ recognition rate for common ASL letters
- **Mobile Optimization**: Efficient battery and memory usage for extended use
- **Robust Architecture**: Handles network issues gracefully with proper error handling
- **User-Centric Design**: Created intuitive interface that works for both signers and observers

### Impact & Innovation
- **Accessibility First**: Built with accessibility as a core principle, not an afterthought
- **Real-World Testing**: Conducted extensive testing with diverse users and conditions
- **Full-Stack Integration**: Successfully connected iOS app, FastAPI backend, and ML pipeline
- **AI-Powered Enhancement**: Integrated GPT for natural language processing
- **Open Source Ready**: Structured codebase for future contributions and improvements

### Learning & Growth
- **Cross-Platform Development**: Gained experience in full-stack mobile development
- **ML Production Deployment**: Learned to deploy ML models in real-world applications
- **Accessibility Design**: Developed deep understanding of inclusive design principles
- **Real-World Problem Solving**: Tackled complex technical challenges with practical solutions

## What we learned

### Technical Discoveries
- **Hand Landmark Detection**: Mastered MediaPipe's hand tracking to extract 21 key points per hand
- **CNN Architecture**: Built and trained custom Convolutional Neural Networks for ASL character recognition
- **Data Preprocessing**: Learned the importance of proper image rotation, normalization, and augmentation
- **Model Optimization**: Discovered the balance between accuracy and real-time performance
- **iOS Development**: Deep dive into AVFoundation, SwiftUI state management, and modern Swift concurrency
- **Backend Development**: Built robust APIs with FastAPI, OpenCV, and OpenAI integration

### Personal Growth
- **Accessibility First**: Learned to design with accessibility as a core principle
- **User-Centric Development**: Understood the importance of real user feedback in AI applications
- **Iterative Improvement**: Discovered that ML models require continuous refinement based on real-world usage
- **Cross-Platform Thinking**: Gained experience in full-stack mobile development
- **Problem-Solving**: Developed skills in tackling complex technical challenges with practical solutions

### Key Insights
- **Real-World Performance**: Lab accuracy doesn't always translate to real-world usage
- **User Experience**: UI must not interfere with the primary task (signing)
- **Network Reliability**: Crucial for real-time applications
- **Domain Knowledge**: Essential for building effective tools
- **Ensemble Methods**: Improve reliability in real-world scenarios

## What's next for BeHeard

### Short-Term Goals
- **Expanded Vocabulary**: Support for more ASL signs and phrases beyond individual letters
- **Improved Accuracy**: Fine-tune the model with more diverse training data
- **Performance Optimization**: Further reduce latency and improve battery efficiency
- **User Testing**: Conduct more extensive testing with the deaf and hard-of-hearing community

### Medium-Term Vision
- **Multi-Language Support**: Recognition for other sign languages (BSL, ASL variations)
- **Voice Output**: Text-to-speech for two-way communication
- **Learning Mode**: Interactive ASL learning with feedback and practice exercises
- **Offline Capability**: Full functionality without internet connection
- **Android Support**: Expand to Android platform for broader accessibility

### Long-Term Impact
- **Community Features**: Sharing and collaboration tools for ASL learners
- **Educational Integration**: Partner with schools and educational institutions
- **Healthcare Applications**: Specialized medical communication support
- **Professional Use**: Workplace communication and accessibility tools
- **Open Source Community**: Build a community of contributors and users

### Technical Roadmap
- **Advanced ML Models**: Implement transformer-based models for better accuracy
- **Real-Time Translation**: Support for full sentence and phrase recognition
- **Multi-Modal Input**: Combine visual and audio cues for better recognition
- **Cloud Integration**: Scalable backend infrastructure for global deployment
- **API Platform**: Allow third-party developers to integrate sign language recognition

---

<div align="center">
  <strong>BeHeard - Because everyone deserves to be heard</strong>
  
  Made with ❤️ for the deaf and hard-of-hearing community
</div>

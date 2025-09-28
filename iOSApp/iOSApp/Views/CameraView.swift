import SwiftUI
import AVFoundation
import Foundation
import UIKit

// MARK: - Main Camera View
struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showingSettings = false
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                // Top Bar
                HStack {
                    Button(action: { showingSettings.toggle() }) {
                        Image(systemName: "line.3.horizontal")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(8)
                    }
                    
                    Spacer()
                    
                    HStack(spacing: 8) {
                        if cameraManager.isSessionRunning {
                            Text("📷 Active")
                                .font(.caption)
                                .foregroundColor(.green)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.green.opacity(0.2))
                                .cornerRadius(6)
                        } else {
                            Text("⏸️ Inactive")
                                .font(.caption)
                                .foregroundColor(.red)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.red.opacity(0.2))
                                .cornerRadius(6)
                        }
                        
                        Button(action: { cameraManager.switchCamera() }) {
                            Image(systemName: "camera.rotate")
                                .font(.title2)
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(8)
                        }
                        
                        if let _ = cameraManager.error {
                            Text("⚠️")
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
                
                // Camera Preview
                CameraPreviewView(cameraManager: cameraManager)
                    .frame(height: geometry.size.height * 2/3)
                    .clipped()
                    .cornerRadius(12)
                    .overlay(
                        // Top left corner - Reset and GPT buttons
                        VStack(spacing: 8) {
                            Button(action: {
                                cameraManager.clearOutputText()
                            }) {
                                Text("RESET")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.black)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.white.opacity(0.9))
                                    .cornerRadius(6)
                            }
                            
                            Button(action: {
                                cameraManager.callGPTAPI()
                            }) {
                                Text("GPT")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.black)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.blue.opacity(0.9))
                                    .cornerRadius(6)
                            }
                        }
                        .padding(.top, 8)
                        .padding(.leading, 8),
                        alignment: .topLeading
                    )
                    .overlay(
                        // Top right corner info
                        VStack(alignment: .trailing, spacing: 4) {
                            Text("Current: \(cameraManager.framePrediction.uppercased())")
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                            
                            Text("Frames: \(cameraManager.frameCount)")
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                        }
                        .padding(.top, 8)
                        .padding(.trailing, 8),
                        alignment: .topTrailing
                    )
                    .onTapGesture {
                        cameraManager.toggleCameraSession()
                    }
                
                // Status / Output
                VStack(spacing: 0) {
                    // Main prediction display - takes full bottom space
                    ScrollView {
                        ScrollViewReader { proxy in
                            Text(cameraManager.finalOutputText.isEmpty ? "Currently analyzing sign language gesture..." : cameraManager.finalOutputText)
                                .onChange(of: cameraManager.finalOutputText) { newValue in
                                    print("DEBUG: Text display updated to: '\(newValue)'")
                                    print("DEBUG: Camera session running: \(cameraManager.isSessionRunning)")
                                }
                                .font(.title)
                                .fontWeight(.semibold)
                                .foregroundColor(.blue)
                                .multilineTextAlignment(.leading)
                                .lineLimit(nil) // allow wrapping
                                .frame(maxWidth: .infinity, alignment: .leading) // fill width but left-align
                                .padding(30)
                                .id("textContent")
                                .onChange(of: cameraManager.finalOutputText) { _ in
                                    // Auto-scroll to bottom when text changes
                                    withAnimation(.easeOut(duration: 0.3)) {
                                        proxy.scrollTo("textContent", anchor: .bottom)
                                    }
                                }
                        }
                    }
                    .background(Color.black)
                }
                .frame(height: geometry.size.height * 1/3)
            }
        }
        .onAppear {
            Task {
                await cameraManager.setupCamera()
                cameraManager.startSession()
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
    }
}

// MARK: - Camera Manager
class CameraManager: NSObject, ObservableObject {
    @Published var isSessionRunning = false
    @Published var error: CameraError?
    @Published var testResult = "Waiting for test..."
    @Published var isProcessingFrame = false
    @Published var frameCount = 0
    
    // Prediction state variables (like React useState)
    @Published var framePrediction = "unknown"
    @Published var status = "waiting"
    @Published var predictionText = "No prediction yet..."
    
    // String storage system for concatenating characters
    @Published var finalOutputText = ""
    private let maxTextLength = 200
    
    // Buffer system for improving prediction accuracy
    private var predictionBuffer: [String] = []
    private let bufferSize = 20
    
    // GPT processing on camera stop
    private var rawString = "" // Raw accumulated string from predictions
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var currentCameraPosition: AVCaptureDevice.Position = .back
    
    private let baseURL = "http://100.64.9.182:8001"
    private var lastProcessTime = Date()
    private let processingInterval: TimeInterval = 0.05
    private var lastTapTime = Date()
    private let tapDebounceInterval: TimeInterval = 1.0
    
    enum CameraError: Error, LocalizedError {
        case permissionDenied, setupFailed, sessionNotRunning
        var errorDescription: String? {
            switch self {
            case .permissionDenied: return "Camera permission denied."
            case .setupFailed: return "Failed to setup camera."
            case .sessionNotRunning: return "Camera session not running."
            }
        }
    }
    
    // Setup
    func setupCamera() async {
        guard await requestPermission() else {
            await MainActor.run { error = .permissionDenied }
            return
        }
        
        sessionQueue.async {
            self.session.beginConfiguration()
            if self.session.canSetSessionPreset(.high) {
                self.session.sessionPreset = .high
            }
            
            guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: self.currentCameraPosition),
                  let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
                  self.session.canAddInput(videoInput) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async { self.error = .setupFailed }
                return
            }
            self.session.addInput(videoInput)
            
            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
                self.videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ]
            }
            
            self.session.commitConfiguration()
            DispatchQueue.main.async { self.error = nil }
        }
    }
    
    // Start
    func startSession() {
        print("DEBUG: startSession called")
        sessionQueue.async {
            if !self.session.isRunning {
                print("DEBUG: Starting camera session")
                self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)
                self.session.startRunning()
                DispatchQueue.main.async {
                    self.isSessionRunning = true
                    self.testResult = "Camera active - sending frames..."
                    self.frameCount = 0
                    
                    // Reset text when starting camera
                    self.finalOutputText = ""
                    self.rawString = ""
                    self.predictionBuffer.removeAll()
                    
                    print("DEBUG: Camera session started, isSessionRunning: \(self.isSessionRunning)")
                    print("DEBUG: Reset text for new session")
                }
            } else {
                print("DEBUG: Camera session already running")
            }
        }
    }
    
    // Stop
    func stopSession() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.videoOutput.setSampleBufferDelegate(nil, queue: nil)
            DispatchQueue.main.async {
                self.isSessionRunning = false
                self.isProcessingFrame = false
                self.testResult = "Camera stopped"
                
                print("DEBUG: Camera stopped, raw string is: '\(self.rawString)'")
                print("DEBUG: About to call GPT processing, isSessionRunning: \(self.isSessionRunning)")
                
                // Process raw string with GPT when camera stops
                self.processWithGPTOnCameraStop()
            }
        }
    }
    
    // Switch Camera
    func switchCamera() {
        sessionQueue.async {
            let newPosition: AVCaptureDevice.Position = self.currentCameraPosition == .front ? .back : .front
            guard let newCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition),
                  let newVideoInput = try? AVCaptureDeviceInput(device: newCamera) else { return }
            
            self.session.beginConfiguration()
            if let currentInput = self.session.inputs.first {
                self.session.removeInput(currentInput)
            }
            if self.session.canAddInput(newVideoInput) {
                self.session.addInput(newVideoInput)
                self.currentCameraPosition = newPosition
            }
            self.session.commitConfiguration()
            
            if let connection = self.videoOutput.connection(with: .video) {
                connection.isVideoMirrored = (newPosition == .front)
                connection.videoOrientation = .portrait
            }
        }
    }
    
    private func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }
    
    var captureSession: AVCaptureSession { session }
    
    func toggleCameraSession() {
        let now = Date()
        if now.timeIntervalSince(lastTapTime) < tapDebounceInterval { return }
        lastTapTime = now
        if session.isRunning { stopSession() } else { startSession() }
    }
    
    // Method to handle character concatenation with buffer system
    private func addCharacterToOutput(_ character: String) {
        // Ignore "unknown" responses
        if character.lowercased() == "unknown" {
            return
        }
        
        // Add character to buffer
        predictionBuffer.append(character)
        
        // If buffer is full, process it
        if predictionBuffer.count >= bufferSize {
            let mostFrequentChar = getMostFrequentCharacter(from: predictionBuffer)
            
            // Add the most frequent character to the raw string
            rawString += mostFrequentChar
            
            // Remove consecutive duplicate characters from raw string
            rawString = removeConsecutiveDuplicates(rawString)
            
            // If raw string exceeds 200 characters, remove the first character (circular buffer)
            if rawString.count > maxTextLength {
                rawString = String(rawString.dropFirst())
            }
            
            print("DEBUG: Raw string updated to: '\(rawString)'")
            
            // Clear the buffer for next batch
            predictionBuffer.removeAll()
        }
    }
    
    // Helper method to find the most frequent character in the buffer
    private func getMostFrequentCharacter(from buffer: [String]) -> String {
        var frequencyCount: [String: Int] = [:]
        
        // Count frequency of each character
        for char in buffer {
            frequencyCount[char, default: 0] += 1
        }
        
        // Find the character with highest frequency
        let mostFrequent = frequencyCount.max { $0.value < $1.value }
        return mostFrequent?.key ?? buffer.last ?? ""
    }
    
    // Helper method to remove consecutive duplicate characters
    private func removeConsecutiveDuplicates(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        
        var result = ""
        var lastChar: Character? = nil
        
        for char in text {
            if char.lowercased() != lastChar?.lowercased() {
                result.append(char)
                lastChar = char
            }
        }
        
        return result
    }
    
    // Method to clear the output text
    func clearOutputText() {
        finalOutputText = ""
        rawString = ""
        predictionBuffer.removeAll()
    }
    
    // Method to process with GPT when camera stops
    private func processWithGPTOnCameraStop() {
        print("DEBUG: processWithGPTOnCameraStop called")
        print("DEBUG: rawString.isEmpty: \(rawString.isEmpty)")
        print("DEBUG: rawString content: '\(rawString)'")
        
        guard !rawString.isEmpty else {
            print("DEBUG: No raw string to process with GPT")
            return
        }
        
        print("DEBUG: Processing raw string with GPT on camera stop: '\(rawString)'")
        
        guard let url = URL(string: "\(baseURL)/process_text/") else {
            print("DEBUG: Invalid URL for GPT API")
            // Fallback to showing raw string
            finalOutputText = rawString
            print("DEBUG: Fallback due to invalid URL - showing raw string: '\(rawString)'")
            print("DEBUG: isSessionRunning: \(isSessionRunning)")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = ["text": rawString]
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        
        print("DEBUG: Sending GPT request with raw string: '\(rawString)'")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                print("DEBUG: GPT API response received")
                
                if let error = error {
                    print("DEBUG: GPT API request error: \(error)")
                    // Fallback to showing raw string
                    self.finalOutputText = self.rawString
                    print("DEBUG: Fallback - showing raw string: '\(self.rawString)'")
                    return
                }
                
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    print("DEBUG: Invalid response from GPT API")
                    // Fallback to showing raw string
                    self.finalOutputText = self.rawString
                    print("DEBUG: Fallback - showing raw string: '\(self.rawString)'")
                    return
                }
                
                print("DEBUG: GPT API response: \(json)")
                
                if let processedText = json["prose"] as? String {
                    print("DEBUG: GPT processed text: '\(processedText)'")
                    // Replace finalOutputText with GPT result
                    self.finalOutputText = processedText
                    print("DEBUG: Updated finalOutputText to: '\(processedText)'")
                    print("DEBUG: isSessionRunning: \(self.isSessionRunning)")
                    print("DEBUG: finalOutputText is now: '\(self.finalOutputText)'")
                } else {
                    // Fallback to showing raw string
                    self.finalOutputText = self.rawString
                    print("DEBUG: No prose in response - showing raw string: '\(self.rawString)'")
                    print("DEBUG: Available keys in response: \(Array(json.keys))")
                    print("DEBUG: isSessionRunning: \(self.isSessionRunning)")
                    print("DEBUG: finalOutputText is now: '\(self.finalOutputText)'")
                }
            }
        }.resume()
    }
    
    // Method to call GPT API with test input
    func callGPTAPI() {
        print("DEBUG: Calling GPT API with test input")
        
        guard let url = URL(string: "\(baseURL)/process_text/") else {
            print("DEBUG: Invalid URL for GPT API")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = ["text": "thsgoodday"]
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        
        print("DEBUG: Sending GPT request with body: \(requestBody)")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("DEBUG: GPT API request error: \(error)")
                    return
                }
                
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    print("DEBUG: Invalid response from GPT API")
                    return
                }
                
                print("DEBUG: GPT API response: \(json)")
                
                if let processedText = json["prose"] as? String {
                    print("DEBUG: GPT processed text: '\(processedText)'")
                    // Update the display with the processed text
                    self.finalOutputText = processedText
                }
            }
        }.resume()
    }
}

// MARK: - Camera Preview
struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    func makeUIView(context: Context) -> CameraPreview {
        let preview = CameraPreview()
        preview.setupPreviewLayer(with: cameraManager.captureSession)
        return preview
    }
    func updateUIView(_ uiView: CameraPreview, context: Context) {}
}

class CameraPreview: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    func setupPreviewLayer(with session: AVCaptureSession) {
        let previewLayer = layer as! AVCaptureVideoPreviewLayer
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspectFill
    }
}

// MARK: - Settings
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                VStack(spacing: 8) {
                    Image(systemName: "gearshape.fill").font(.system(size: 50)).foregroundColor(.blue)
                    Text("Settings").font(.largeTitle).fontWeight(.bold)
                    Text("Configure your sign language app").font(.subheadline).foregroundColor(.secondary)
                }
                Spacer()
                VStack(spacing: 16) {
                    SettingsRow(icon: "camera.fill", title: "Camera Settings", description: "Configure camera preferences")
                    SettingsRow(icon: "brain.head.profile", title: "Model Settings", description: "Adjust AI model parameters")
                    SettingsRow(icon: "textformat", title: "Language Settings", description: "Choose output language")
                    SettingsRow(icon: "bell.fill", title: "Notifications", description: "Manage app notifications")
                    SettingsRow(icon: "questionmark.circle.fill", title: "Help & Support", description: "Get help and support")
                }
                Spacer()
                Text("Sign Language Recognition App").font(.caption).foregroundColor(.secondary).padding(.bottom, 20)
            }
            .padding()
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { Button("Done") { dismiss() } } }
        }
    }
}

struct SettingsRow: View {
    let icon: String, title: String, description: String
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon).font(.title2).foregroundColor(.blue).frame(width: 30)
            VStack(alignment: .leading) {
                Text(title).font(.headline)
                Text(description).font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Image(systemName: "chevron.right").font(.caption).foregroundColor(.secondary)
        }
        .padding().background(Color(.systemGray6)).cornerRadius(12)
    }
}

// MARK: - Delegate (Frame → JPEG → Base64 → Backend)
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        print("DEBUG: captureOutput called, isSessionRunning: \(isSessionRunning)")
        guard isSessionRunning else { 
            print("DEBUG: Session not running, returning")
            return 
        }
        let now = Date()
        let timeSinceLastProcess = now.timeIntervalSince(lastProcessTime)
        print("DEBUG: Time since last process: \(timeSinceLastProcess), interval: \(processingInterval)")
        if timeSinceLastProcess >= processingInterval {
            print("DEBUG: Processing frame for backend")
            lastProcessTime = now
            processFrameForBackend(sampleBuffer)
        }
    }
    
    private func processFrameForBackend(_ sampleBuffer: CMSampleBuffer) {
        print("DEBUG: processFrameForBackend called")
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { 
            print("DEBUG: Failed to get pixel buffer")
            return 
        }
        
        DispatchQueue.main.async {
            self.isProcessingFrame = true
            self.frameCount += 1
            print("DEBUG: Frame count: \(self.frameCount)")
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        
        // Create UIImage with correct orientation to prevent rotation
        // For rear camera, use .up orientation; for front camera, use .upMirrored
        let orientation: UIImage.Orientation = currentCameraPosition == .back ? .up : .upMirrored
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
        
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.4) else { return }
        sendFrameToBackend(imageData: jpegData)
    }
    
    private func sendFrameToBackend(imageData: Data) {
        print("DEBUG: sendFrameToBackend called with imageData size: \(imageData.count)")
        guard let url = URL(string: "\(baseURL)/predict_sign/") else { 
            print("DEBUG: Failed to create URL")
            return 
        }
        print("DEBUG: Sending request to: \(url)")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let base64String = imageData.base64EncodedString()
        let requestBody: [String: Any] = ["image": base64String]
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        
        URLSession.shared.dataTask(with: request) { data, _, error in
            DispatchQueue.main.async {
                self.isProcessingFrame = false
                if let error = error {
                    self.status = "error"
                    self.framePrediction = "error"
                    self.testResult = "Error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    self.status = "error"
                    self.framePrediction = "error"
                    self.testResult = "Invalid response"
                    return
                }
                
                // Parse prediction results (like React useState updates)
                if let responseStatus = json["status"] as? String {
                    // Update status
                    self.status = responseStatus
                    
                    if responseStatus == "success" {
                        // Update frame prediction
                        if let framePred = json["frame_prediction"] as? String {
                            self.framePrediction = framePred
                            // Add character to final output text (ignores "unknown")
                            self.addCharacterToOutput(framePred)
                        }
                        
                        // Update test result for debugging
                        if let message = json["message"] as? String {
                            self.testResult = message
                        }
                    } else {
                        self.framePrediction = "error"
                        self.testResult = "Prediction failed: \(responseStatus)"
                    }
                } else {
                    self.status = "error"
                    self.framePrediction = "error"
                    self.testResult = "Invalid response"
                }
            }
        }.resume()
    }
}

#Preview {
    NavigationView { CameraView() }
}

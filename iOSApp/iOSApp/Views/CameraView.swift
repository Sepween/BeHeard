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
                    .onTapGesture {
                        cameraManager.toggleCameraSession()
                    }
                
                // Status / Output
                VStack(spacing: 10) {
                    Text(cameraManager.testResult)
                        .font(.title2)
                        .fontWeight(.medium)
                        .multilineTextAlignment(.center)
                    
                    Text("Frames sent: \(cameraManager.frameCount)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if cameraManager.isProcessingFrame {
                        ProgressView().scaleEffect(0.8)
                    }
                    
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding()
                .background(Color(.systemGray6))
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
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var currentCameraPosition: AVCaptureDevice.Position = .front
    
    private let baseURL = "http://35.3.45.24:8001"
    private var lastProcessTime = Date()
    private let processingInterval: TimeInterval = 1.0 // send ~1 frame per second
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
            
            guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
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
        sessionQueue.async {
            if !self.session.isRunning {
                self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)
                self.session.startRunning()
                DispatchQueue.main.async {
                    self.isSessionRunning = true
                    self.testResult = "Camera active - sending frames..."
                    self.frameCount = 0
                }
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
        guard isSessionRunning else { return }
        let now = Date()
        if now.timeIntervalSince(lastProcessTime) >= processingInterval {
            lastProcessTime = now
            processFrameForBackend(sampleBuffer)
        }
    }
    
    private func processFrameForBackend(_ sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        DispatchQueue.main.async {
            self.isProcessingFrame = true
            self.frameCount += 1
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.4) else { return }
        sendFrameToBackend(imageData: jpegData)
    }
    
    private func sendFrameToBackend(imageData: Data) {
        guard let url = URL(string: "\(baseURL)/test_predict/") else { return }
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
                    self.testResult = "Error: \(error.localizedDescription)"
                    return
                }
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let message = json["message"] as? String {
                    self.testResult = message
                } else {
                    self.testResult = "Invalid response"
                }
            }
        }.resume()
    }
}

#Preview {
    NavigationView { CameraView() }
}

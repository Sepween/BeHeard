import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showingSettings = false
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                // Top Bar - Settings and Camera Controls
                HStack {
                    // Settings Button - Top Left
                    Button(action: {
                        showingSettings.toggle()
                    }) {
                        Image(systemName: "line.3.horizontal")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(8)
                    }
                    
                    Spacer()
                    
                    // Camera Controls - Top Right
                    HStack(spacing: 8) {
                        // Camera Status
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
                        
                        // Camera Flip Button
                        Button(action: {
                            cameraManager.switchCamera()
                        }) {
                            Image(systemName: "camera.rotate")
                                .font(.title2)
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(8)
                        }
                        
                        if let error = cameraManager.error {
                            Text("⚠️")
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
                
                // Camera Preview - Takes 2/3 of screen
                CameraPreviewView(cameraManager: cameraManager)
                    .frame(height: geometry.size.height * 2/3)
                    .clipped()
                    .onTapGesture {
                        // Toggle camera on tap
                        if cameraManager.isSessionRunning {
                            cameraManager.stopSession()
                        } else {
                            cameraManager.startSession()
                        }
                    }
                
                // Text Box - Takes 1/3 of screen
                VStack {
                    Text("This is a placeholder.")
                        .font(.title2)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding()
                        .background(Color(.systemGray6))
                }
                .frame(height: geometry.size.height * 1/3)
                .background(Color(.systemBackground))
            }
        }
        .onAppear {
            Task {
                await cameraManager.setupCamera()
                cameraManager.startSession() // Auto-start camera
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
    }
}

// Camera Manager
class CameraManager: NSObject, ObservableObject {
    @Published var isSessionRunning = false
    @Published var error: CameraError?
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var currentCameraPosition: AVCaptureDevice.Position = .front
    
    enum CameraError: Error, LocalizedError {
        case permissionDenied
        case setupFailed
        case sessionNotRunning
        
        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Camera permission denied. Please enable in Settings."
            case .setupFailed:
                return "Failed to setup camera"
            case .sessionNotRunning:
                return "Camera session not running"
            }
        }
    }
    
    override init() {
        super.init()
    }
    
    func setupCamera() async {
        // Request permission
        guard await requestPermission() else {
            await MainActor.run {
                error = .permissionDenied
            }
            return
        }
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.session.beginConfiguration()
            
            // Set session preset
            if self.session.canSetSessionPreset(.high) {
                self.session.sessionPreset = .high
            }
            
            // Add video input
            guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
                  let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
                  self.session.canAddInput(videoInput) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async {
                    self.error = .setupFailed
                }
                return
            }
            
            self.session.addInput(videoInput)
            
            // Add video output
            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
                
                // Set video connection
                if let connection = self.videoOutput.connection(with: .video) {
                    if connection.isVideoMirroringSupported {
                        connection.isVideoMirrored = true
                    }
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                }
            }
            
            self.session.commitConfiguration()
            
            DispatchQueue.main.async {
                self.error = nil
            }
        }
    }
    
    func startSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
                DispatchQueue.main.async {
                    self.isSessionRunning = true
                }
            }
        }
    }
    
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
                DispatchQueue.main.async {
                    self.isSessionRunning = false
                }
            }
        }
    }
    
    private func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }
    
    var captureSession: AVCaptureSession {
        return session
    }
    
    /// Switch between front and back camera
    func switchCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            self.session.beginConfiguration()
            
            // Remove current input
            if let currentInput = self.session.inputs.first {
                self.session.removeInput(currentInput)
            }
            
            // Switch camera position
            self.currentCameraPosition = self.currentCameraPosition == .front ? .back : .front
            
            // Add new input
            guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: self.currentCameraPosition),
                  let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
                  self.session.canAddInput(videoInput) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async {
                    self.error = .setupFailed
                }
                return
            }
            
            self.session.addInput(videoInput)
            
            // Update video connection orientation and mirroring
            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = (self.currentCameraPosition == .front)
                }
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            
            self.session.commitConfiguration()
        }
    }
}

// Camera Preview View
struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    func makeUIView(context: Context) -> CameraPreview {
        let preview = CameraPreview()
        preview.setupPreviewLayer(with: cameraManager.captureSession)
        return preview
    }
    
    func updateUIView(_ uiView: CameraPreview, context: Context) {
        // Update if needed
    }
}

class CameraPreview: UIView {
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    func setupPreviewLayer(with session: AVCaptureSession) {
        previewLayer = layer as? AVCaptureVideoPreviewLayer
        previewLayer?.session = session
        previewLayer?.videoGravity = .resizeAspectFill
    }
}

// Settings View
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)
                    
                    Text("Settings")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Configure your sign language app")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 20)
                
                Spacer()
                
                // Settings Options (Placeholders)
                VStack(spacing: 16) {
                    SettingsRow(icon: "camera.fill", title: "Camera Settings", description: "Configure camera preferences")
                    SettingsRow(icon: "brain.head.profile", title: "Model Settings", description: "Adjust AI model parameters")
                    SettingsRow(icon: "textformat", title: "Language Settings", description: "Choose output language")
                    SettingsRow(icon: "bell.fill", title: "Notifications", description: "Manage app notifications")
                    SettingsRow(icon: "questionmark.circle.fill", title: "Help & Support", description: "Get help and support")
                }
                
                Spacer()
                
                // Footer
                Text("Sign Language Recognition App")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 20)
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// Settings Row Component
struct SettingsRow: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onTapGesture {
            // Placeholder action
            print("Tapped: \(title)")
        }
    }
}

#Preview {
    NavigationView {
        CameraView()
    }
}

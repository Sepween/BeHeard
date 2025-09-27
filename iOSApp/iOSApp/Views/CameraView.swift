import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        VStack {
            // Camera Preview
            CameraPreviewView(cameraManager: cameraManager)
                .aspectRatio(4/3, contentMode: .fit)
                .clipped()
                .cornerRadius(12)
                .padding()
            
            // Controls
            VStack(spacing: 20) {
                if cameraManager.isSessionRunning {
                    Text("Camera is running! 📷")
                        .font(.headline)
                        .foregroundColor(.green)
                } else {
                    Text("Camera not running")
                        .font(.headline)
                        .foregroundColor(.red)
                }
                
                if let error = cameraManager.error {
                    Text("Error: \(error.localizedDescription)")
                        .font(.caption)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                }
                
                Button(action: {
                    if cameraManager.isSessionRunning {
                        cameraManager.stopSession()
                    } else {
                        cameraManager.startSession()
                    }
                }) {
                    HStack {
                        Image(systemName: cameraManager.isSessionRunning ? "stop.circle.fill" : "play.circle.fill")
                        Text(cameraManager.isSessionRunning ? "Stop Camera" : "Start Camera")
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(cameraManager.isSessionRunning ? Color.red : Color.blue)
                    .cornerRadius(10)
                }
            }
            .padding()
            
            Spacer()
        }
        .navigationTitle("Camera")
        .onAppear {
            Task {
                await cameraManager.setupCamera()
            }
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

#Preview {
    NavigationView {
        CameraView()
    }
}

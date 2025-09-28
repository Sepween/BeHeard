from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import base64
import cv2

app = FastAPI(title="Simple Sign Language Backend", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model for Base64-encoded image
class ImageRequest(BaseModel):
    image: str  # Base64-encoded string

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simple Sign Language Backend is running with Base64 input!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/test_predict/")
async def test_predict(request: ImageRequest):
    """
    Test endpoint that receives Base64 image and returns confirmation
    """
    try:
        # Decode base64 -> raw bytes
        img_bytes = base64.b64decode(request.image)

        # Convert bytes -> numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image using OpenCV (BGR by default)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Debug info
        print(f"INFO: Received image with shape: {img_rgb.shape}")
        print(f"INFO: First pixel RGB values: {img_rgb[0,0].tolist()}")

        return {
            "status": "success",
            "message": "Image received",
            "received_shape": list(img_rgb.shape),  # [height, width, channels]
            "total_pixels": int(img_rgb.size),
            "dtype": str(img_rgb.dtype)
        }

    except Exception as e:
        print(f"ERROR: Failed to process image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Sign Language Backend...")
    print("Server will be available at: http://localhost:8001")
    print("Health check: http://localhost:8001/health")
    print("Test endpoint: http://localhost:8001/test_predict/")
    uvicorn.run(app, host="0.0.0.0", port=8001)

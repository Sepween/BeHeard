from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="iOS App API", version="1.0.0")

# Enable CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your iOS app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class User(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str] = None

class CreateUser(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

class Message(BaseModel):
    id: int
    content: str
    user_id: int
    timestamp: str

class CreateMessage(BaseModel):
    content: str
    user_id: int

class SignLanguageSession(BaseModel):
    id: int
    user_id: int
    start_time: str
    end_time: Optional[str] = None
    total_predictions: int = 0
    accuracy_score: Optional[float] = None

class CreateSignLanguageSession(BaseModel):
    user_id: int

class SignLanguagePrediction(BaseModel):
    id: int
    session_id: int
    predicted_letter: str
    confidence: float
    timestamp: str
    landmarks_data: Optional[str] = None  # JSON string of landmarks

class CreateSignLanguagePrediction(BaseModel):
    session_id: int
    predicted_letter: str
    confidence: float
    landmarks_data: Optional[str] = None

# In-memory storage (replace with database in production)
users_db = []
messages_db = []
sign_language_sessions_db = []
sign_language_predictions_db = []
user_id_counter = 1
message_id_counter = 1
session_id_counter = 1
prediction_id_counter = 1

@app.get("/")
async def root():
    return {"message": "Welcome to iOS App API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# User endpoints
@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = next((u for u in users_db if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users", response_model=User)
async def create_user(user: CreateUser):
    global user_id_counter
    new_user = {
        "id": user_id_counter,
        "name": user.name,
        "email": user.email,
        "phone": user.phone
    }
    users_db.append(new_user)
    user_id_counter += 1
    return new_user

@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user: CreateUser):
    user_index = next((i for i, u in enumerate(users_db) if u["id"] == user_id), None)
    if user_index is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    users_db[user_index].update({
        "name": user.name,
        "email": user.email,
        "phone": user.phone
    })
    return users_db[user_index]

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    global users_db
    users_db = [u for u in users_db if u["id"] != user_id]
    return {"message": "User deleted successfully"}

# Message endpoints
@app.get("/messages", response_model=List[Message])
async def get_messages():
    return messages_db

@app.get("/messages/{message_id}", response_model=Message)
async def get_message(message_id: int):
    message = next((m for m in messages_db if m["id"] == message_id), None)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message

@app.post("/messages", response_model=Message)
async def create_message(message: CreateMessage):
    global message_id_counter
    from datetime import datetime
    
    new_message = {
        "id": message_id_counter,
        "content": message.content,
        "user_id": message.user_id,
        "timestamp": datetime.now().isoformat()
    }
    messages_db.append(new_message)
    message_id_counter += 1
    return new_message

@app.get("/users/{user_id}/messages", response_model=List[Message])
async def get_user_messages(user_id: int):
    user_messages = [m for m in messages_db if m["user_id"] == user_id]
    return user_messages

# Sign Language Session endpoints
@app.get("/sign-language/sessions", response_model=List[SignLanguageSession])
async def get_sign_language_sessions():
    return sign_language_sessions_db

@app.get("/sign-language/sessions/{session_id}", response_model=SignLanguageSession)
async def get_sign_language_session(session_id: int):
    session = next((s for s in sign_language_sessions_db if s["id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.post("/sign-language/sessions", response_model=SignLanguageSession)
async def create_sign_language_session(session: CreateSignLanguageSession):
    global session_id_counter
    from datetime import datetime
    
    new_session = {
        "id": session_id_counter,
        "user_id": session.user_id,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_predictions": 0,
        "accuracy_score": None
    }
    sign_language_sessions_db.append(new_session)
    session_id_counter += 1
    return new_session

@app.put("/sign-language/sessions/{session_id}", response_model=SignLanguageSession)
async def end_sign_language_session(session_id: int):
    from datetime import datetime
    
    session_index = next((i for i, s in enumerate(sign_language_sessions_db) if s["id"] == session_id), None)
    if session_index is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate accuracy score based on predictions
    session_predictions = [p for p in sign_language_predictions_db if p["session_id"] == session_id]
    if session_predictions:
        avg_confidence = sum(p["confidence"] for p in session_predictions) / len(session_predictions)
        sign_language_sessions_db[session_index]["accuracy_score"] = avg_confidence
    
    sign_language_sessions_db[session_index]["end_time"] = datetime.now().isoformat()
    return sign_language_sessions_db[session_index]

# Sign Language Prediction endpoints
@app.get("/sign-language/predictions", response_model=List[SignLanguagePrediction])
async def get_sign_language_predictions():
    return sign_language_predictions_db

@app.get("/sign-language/predictions/{prediction_id}", response_model=SignLanguagePrediction)
async def get_sign_language_prediction(prediction_id: int):
    prediction = next((p for p in sign_language_predictions_db if p["id"] == prediction_id), None)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@app.post("/sign-language/predictions", response_model=SignLanguagePrediction)
async def create_sign_language_prediction(prediction: CreateSignLanguagePrediction):
    global prediction_id_counter
    from datetime import datetime
    
    new_prediction = {
        "id": prediction_id_counter,
        "session_id": prediction.session_id,
        "predicted_letter": prediction.predicted_letter,
        "confidence": prediction.confidence,
        "timestamp": datetime.now().isoformat(),
        "landmarks_data": prediction.landmarks_data
    }
    sign_language_predictions_db.append(new_prediction)
    
    # Update session prediction count
    session_index = next((i for i, s in enumerate(sign_language_sessions_db) if s["id"] == prediction.session_id), None)
    if session_index is not None:
        sign_language_sessions_db[session_index]["total_predictions"] += 1
    
    prediction_id_counter += 1
    return new_prediction

@app.get("/sign-language/sessions/{session_id}/predictions", response_model=List[SignLanguagePrediction])
async def get_session_predictions(session_id: int):
    session_predictions = [p for p in sign_language_predictions_db if p["session_id"] == session_id]
    return session_predictions

@app.get("/sign-language/users/{user_id}/sessions", response_model=List[SignLanguageSession])
async def get_user_sign_language_sessions(user_id: int):
    user_sessions = [s for s in sign_language_sessions_db if s["user_id"] == user_id]
    return user_sessions

@app.get("/sign-language/analytics/summary")
async def get_sign_language_analytics():
    """Get summary analytics for sign language recognition"""
    total_sessions = len(sign_language_sessions_db)
    total_predictions = len(sign_language_predictions_db)
    
    if total_predictions > 0:
        avg_confidence = sum(p["confidence"] for p in sign_language_predictions_db) / total_predictions
        
        # Most common predicted letters
        letter_counts = {}
        for prediction in sign_language_predictions_db:
            letter = prediction["predicted_letter"]
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        most_common_letters = sorted(letter_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    else:
        avg_confidence = 0.0
        most_common_letters = []
    
    return {
        "total_sessions": total_sessions,
        "total_predictions": total_predictions,
        "average_confidence": round(avg_confidence, 3),
        "most_common_letters": [{"letter": letter, "count": count} for letter, count in most_common_letters]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# FastAPI Backend

A FastAPI backend server that provides RESTful API endpoints for the iOS application.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python run.py
   ```

3. **Access the API:**
   - Server: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

## API Documentation

### Users Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/users` | Get all users |
| GET | `/users/{id}` | Get user by ID |
| POST | `/users` | Create new user |
| PUT | `/users/{id}` | Update user |
| DELETE | `/users/{id}` | Delete user |

### Messages Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/messages` | Get all messages |
| GET | `/messages/{id}` | Get message by ID |
| POST | `/messages` | Create new message |
| GET | `/users/{id}/messages` | Get messages by user |

### Request/Response Examples

#### Create User
```json
POST /users
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890"
}
```

#### Create Message
```json
POST /messages
{
  "content": "Hello, world!",
  "user_id": 1
}
```

## Development

### Running in Development Mode
```bash
python run.py
```

### Running with Uvicorn Directly
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## Data Storage

Currently uses in-memory storage. Data will be lost when the server restarts.

To persist data, integrate with:
- SQLite: `pip install sqlalchemy`
- PostgreSQL: `pip install sqlalchemy psycopg2`
- MongoDB: `pip install motor`

## CORS Configuration

CORS is enabled for all origins in development. For production, update the CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

# iOS Sign Language Recognition App with FastAPI Backend

A complete iOS application with **real-time sign language recognition** that communicates with a FastAPI backend server. This project includes both the iOS Swift app with camera-based gesture recognition and the Python FastAPI backend for data storage and analytics.

## Project Structure

```
mhacks25/
├── backend/                 # FastAPI Backend
│   ├── main.py             # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   ├── run.py             # Server runner script
│   └── README.md          # Backend documentation
├── iOSApp/                 # iOS Application
│   ├── iOSApp.xcodeproj/   # Xcode project file
│   └── iOSApp/             # Source code
│       ├── iOSAppApp.swift # App entry point
│       ├── ContentView.swift # Main view
│       ├── Models/         # Data models
│       │   ├── User.swift
│       │   └── Message.swift
│       ├── Views/          # SwiftUI views
│       │   ├── UserListView.swift
│       │   └── MessageListView.swift
│       ├── Network/        # API client
│       │   └── APIClient.swift
│       ├── Assets.xcassets # App assets
│       └── Preview Content/ # SwiftUI previews
└── README.md              # This file
```

## Features

### Backend (FastAPI)
- RESTful API with CRUD operations for users and messages
- CORS enabled for iOS app communication
- In-memory data storage (easily replaceable with database)
- Automatic API documentation at `/docs`
- Health check endpoint

### iOS App (SwiftUI)
- **Real-time Sign Language Recognition** with camera integration
- Tab-based navigation with Users, Messages, and Sign Language sections
- Modern SwiftUI interface with async/await
- Full CRUD operations for users
- Message creation and display
- **Hand landmark detection** using Vision framework
- **Core ML integration** for gesture recognition
- Error handling and loading states
- Real-time data updates

## API Endpoints

### Users
- `GET /users` - Get all users
- `GET /users/{id}` - Get user by ID
- `POST /users` - Create new user
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

### Messages
- `GET /messages` - Get all messages
- `GET /messages/{id}` - Get message by ID
- `POST /messages` - Create new message
- `GET /users/{id}/messages` - Get messages by user

### Sign Language Recognition
- `GET /sign-language/sessions` - Get all recognition sessions
- `POST /sign-language/sessions` - Create new session
- `PUT /sign-language/sessions/{id}` - End session
- `GET /sign-language/predictions` - Get all predictions
- `POST /sign-language/predictions` - Store prediction
- `GET /sign-language/analytics/summary` - Get usage analytics

### System
- `GET /` - Welcome message
- `GET /health` - Health check

## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd mhacks25/backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```bash
   python run.py
   ```

   The server will start on `http://localhost:8000`
   
   You can view the interactive API documentation at `http://localhost:8000/docs`

### iOS App Setup

1. **Open the iOS project:**
   - Open `mhacks25/iOSApp/iOSApp.xcodeproj` in Xcode
   - Make sure you have Xcode 15.0+ installed

2. **Configure the API endpoint:**
   - Open `mhacks25/iOSApp/iOSApp/Network/APIClient.swift`
   - Update the `baseURL` constant to match your backend server URL
   - For local development: `http://localhost:8000`
   - For simulator testing: `http://127.0.0.1:8000`
   - For physical device testing: `http://[YOUR_IP]:8000`

3. **Run the app:**
   - Select your target device (simulator or physical device)
   - Press Cmd+R to build and run

## Configuration Notes

### Network Configuration

**For iOS Simulator:**
- Use `http://localhost:8000` or `http://127.0.0.1:8000`

**For Physical iOS Device:**
- Find your computer's IP address: `ipconfig getifaddr en0` (Mac) or `ipconfig` (Windows)
- Use `http://[YOUR_IP]:8000` in the APIClient.swift file
- Ensure both devices are on the same network

**For Production:**
- Replace the baseURL with your production server URL
- Implement proper authentication if needed
- Add SSL/TLS certificates for HTTPS

### CORS Configuration

The backend is configured to allow all origins (`allow_origins=["*"]`) for development. In production, you should specify your iOS app's bundle identifier or domain.

## Development Tips

### Backend Development
- The API uses in-memory storage, so data will be lost when the server restarts
- To persist data, integrate with a database like SQLite, PostgreSQL, or MongoDB
- Add authentication using JWT tokens or OAuth
- Implement proper error handling and logging

### iOS Development
- The app uses modern Swift concurrency (async/await)
- All network calls are performed on background threads
- UI updates are dispatched to the main thread automatically
- Error states are handled gracefully with user-friendly messages

### Testing
- Test the backend API using the interactive docs at `/docs`
- Use tools like Postman or curl to test endpoints
- Test the iOS app with both simulator and physical devices
- Verify network connectivity between devices

## Next Steps

1. **Database Integration:** Replace in-memory storage with a real database
2. **Authentication:** Add user authentication and authorization
3. **Push Notifications:** Implement real-time notifications
4. **Offline Support:** Add local data caching for offline functionality
5. **Testing:** Add unit tests and UI tests
6. **Deployment:** Deploy backend to cloud services like Heroku, AWS, or DigitalOcean

## Troubleshooting

### Common Issues

**"Network request failed" in iOS app:**
- Check if the backend server is running
- Verify the baseURL in APIClient.swift
- Ensure both devices are on the same network (for physical device testing)
- Check firewall settings

**CORS errors:**
- The backend includes CORS middleware, but verify it's properly configured
- Check browser developer tools for specific CORS error messages

**Build errors in Xcode:**
- Ensure you're using Xcode 15.0 or later
- Clean build folder (Product → Clean Build Folder)
- Check that all Swift files are properly added to the project

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

# Galamsey AI Detector Backend - README
# =====================================

## Overview
This backend server integrates YOLOv12 for real-time illegal mining detection using webcam feeds. It's designed to work with the Galamsey AI Detector frontend and can be deployed on Raspberry Pi 5 for drone-based monitoring.

## Features
- ✅ YOLOv12 model integration
- ✅ Real-time webcam detection
- ✅ Flask API with video streaming
- ✅ Health check endpoints
- ✅ Detection statistics
- ✅ Configurable detection thresholds
- 🔄 Future drone integration (commented)
- 🔄 Future frontend API endpoints (commented)

## Project Structure
```
backend/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── setup.sh           # Setup script
└── README.md          # This file
```

## Quick Start

### 1. Install Dependencies
```bash
cd backend
chmod +x setup.sh
./setup.sh
```

### 2. Start the Server
```bash
source venv/bin/activate
python app.py
```

### 3. Test the Endpoints
- **Health Check**: http://localhost:5000/health
- **Video Feed**: http://localhost:5000/video_feed
- **Statistics**: http://localhost:5000/statistics
- **Configuration**: http://localhost:5000/config

## API Endpoints

### GET /
Root endpoint with API information

### GET /health
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2025-01-14T14:23:00",
  "camera": "connected",
  "model": "loaded",
  "device": "cpu",
  "detection_threshold": 0.5
}
```

### GET /video_feed
Real-time video stream with detections
- MIME type: `multipart/x-mixed-replace; boundary=frame`
- Returns: Live video feed with bounding boxes

### GET /statistics
Detection statistics
```json
{
  "total_detections": 47,
  "illegal_detections": 24,
  "legal_detections": 23,
  "session_duration": "0:15:30",
  "detection_rate": 0.05,
  "illegal_percentage": 51.06
}
```

### GET/POST /config
Configuration management
- GET: Retrieve current settings
- POST: Update detection threshold

## Model Integration

### YOLOv12 Setup
1. Place your YOLOv12 model files in the `yolov12/` directory
2. Supported formats: `.pt`, `.onnx`, `.engine`
3. The system will automatically load the model on startup

### Dataset Integration
1. Place your "illegal mining activities" dataset in the project root
2. Create a `classes.txt` file with class names (optional)
3. Default classes are used if no custom classes are found

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development`, `production`, or `raspberry_pi`
- `MODEL_PATH`: Path to YOLOv12 model directory
- `DATASET_PATH`: Path to dataset directory

### Detection Classes
Default illegal mining indicators:
- excavator, bulldozer, water_pump, wash_plant
- truck, conveyor, drill, mining_equipment
- illegal_site, suspicious_activity

## Raspberry Pi Deployment

### Prerequisites
- Raspberry Pi 5 (i5, 16GB RAM recommended)
- Pi Camera Module
- Ubuntu/Raspberry Pi OS

### Setup Commands
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-pip python3-venv

# Clone and setup
git clone <your-repo>
cd backend
export FLASK_ENV=raspberry_pi
./setup.sh

# Start server
source venv/bin/activate
python app.py
```

### Pi-Specific Optimizations
- Reduced FPS (15 instead of 30)
- Lower resolution (1280x720)
- Optimized threading
- Pi Camera integration

## Future Integrations

### Drone Integration (Commented Out)
The following endpoints are prepared for future drone integration:
- `/drone/connect` - Connect to drone
- `/drone/status` - Monitor drone status
- `/drone/control` - Flight control
- `/gps/coordinates` - GPS tracking

### Frontend Integration (Commented Out)
API endpoints for frontend communication:
- `/api/alerts` - Alert system
- `/api/reports` - Report generation
- `/api/map_data` - Map data
- `/api/export` - Data export

## Troubleshooting

### Camera Issues
```bash
# Check available cameras
ls /dev/video*

# Test camera with OpenCV
python3 -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Model Loading Issues
- Ensure YOLOv12 model files are in the correct directory
- Check file permissions
- Verify model format compatibility

### Performance Issues
- Reduce camera resolution
- Lower detection threshold
- Use GPU acceleration if available
- Optimize for Raspberry Pi hardware

## Development

### Adding New Detection Classes
1. Update `ILLEGAL_CLASSES` in `config.py`
2. Retrain YOLOv12 model with new classes
3. Update class names in dataset

### Custom Model Integration
1. Modify `_load_model()` method in `app.py`
2. Update detection processing logic
3. Test with your specific model format

## License
Environmental AI Solutions - 2025

## Support
For issues and questions, please refer to the project documentation or contact the development team.

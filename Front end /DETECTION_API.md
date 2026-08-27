# Detection Data Transmission API

## Overview
The system automatically sends detection data (location, time, and detected items) to the backend when AI detections occur.

## Data Format

### Detection Payload Structure
```json
{
  "timestamp": "2025-01-14T14:23:45.123Z",
  "location": {
    "lat": 5.603700,
    "lng": -0.187000,
    "altitude": 100.5,
    "heading": 45
  },
  "detections": [
    {
      "type": "excavator",
      "confidence": 85,
      "position": {
        "x": 320,
        "y": 240
      },
      "detected": true
    },
    {
      "type": "pump",
      "confidence": 92,
      "position": {
        "x": 450,
        "y": 180
      },
      "detected": true
    }
  ],
  "metadata": {
    "frameCount": 1234,
    "detectionCount": 2,
    "status": "illegal",
    "geofenceStatus": {
      "insideAllowedArea": false,
      "insideRestrictedArea": true
    }
  }
}
```

## Transmission Methods

### 1. WebSocket (Preferred)
**Format:**
```json
{
  "type": "detection",
  "data": {
    // ... detection payload above
  }
}
```

**WebSocket URL:** `ws://<droneIP>:<dronePort>/ws`

**When sent:**
- Automatically when new detections are received via WebSocket
- Every 5 seconds if detections are present
- Manually via "Send Detection" button

### 2. HTTP API (Fallback)
**Endpoint:** `POST http://<droneIP>:<dronePort>/api/detections`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  // ... detection payload above
}
```

**Response:**
```json
{
  "success": true,
  "message": "Detection data received",
  "id": "detection_1234567890"
}
```

## Detection Types

Currently supported detection types:
- `excavator` - Excavator equipment detected
- `pump` - Water pump detected
- `washplant` - Washing plant detected

## Features

### Automatic Transmission
- Sends detection data automatically every 5 seconds when detections are present
- Only sends if confidence meets threshold (default: 80%)
- Avoids duplicate transmissions using hash-based deduplication

### Manual Transmission
- Click "Send Detection" button in the report panel
- Or call from browser console: `detector.sendCurrentDetectionData()`

### Retry Mechanism
- Failed transmissions are queued in localStorage
- Automatic retry every 30 seconds
- Queued items expire after 1 hour

### Duplicate Prevention
- Uses hash-based detection to avoid sending identical data
- Hash is based on location and detected items

## Backend Implementation Example

### Node.js/Express Example
```javascript
const express = require('express');
const app = express();

app.use(express.json());

// WebSocket server
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
    ws.on('message', (message) => {
        const data = JSON.parse(message);
        
        if (data.type === 'detection') {
            // Handle detection data
            console.log('Received detection:', data.data);
            
            // Save to database
            saveDetectionToDatabase(data.data);
            
            // Send acknowledgment
            ws.send(JSON.stringify({
                type: 'ack',
                message: 'Detection received'
            }));
        }
    });
});

// HTTP API endpoint
app.post('/api/detections', (req, res) => {
    const detectionData = req.body;
    
    // Validate data
    if (!detectionData.timestamp || !detectionData.location || !detectionData.detections) {
        return res.status(400).json({ error: 'Invalid detection data' });
    }
    
    // Save to database
    saveDetectionToDatabase(detectionData);
    
    res.json({
        success: true,
        message: 'Detection data received',
        id: generateDetectionId()
    });
});

function saveDetectionToDatabase(data) {
    // Example: Save to database
    // db.detections.insert({
    //     timestamp: data.timestamp,
    //     location: data.location,
    //     detections: data.detections,
    //     metadata: data.metadata
    // });
    console.log('Saving to database:', data);
}

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

### Python/Flask Example
```python
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/detections', methods=['POST'])
def receive_detection():
    data = request.json
    
    # Validate data
    if not all(key in data for key in ['timestamp', 'location', 'detections']):
        return jsonify({'error': 'Invalid detection data'}), 400
    
    # Save to database
    save_detection_to_database(data)
    
    return jsonify({
        'success': True,
        'message': 'Detection data received',
        'id': generate_detection_id()
    })

@socketio.on('message')
def handle_message(message):
    data = json.loads(message)
    
    if data.get('type') == 'detection':
        detection_data = data.get('data')
        
        # Save to database
        save_detection_to_database(detection_data)
        
        # Send acknowledgment
        emit('response', {
            'type': 'ack',
            'message': 'Detection received'
        })

def save_detection_to_database(data):
    # Example: Save to database
    # db.detections.insert_one({
    #     'timestamp': data['timestamp'],
    #     'location': data['location'],
    #     'detections': data['detections'],
    #     'metadata': data.get('metadata', {})
    # })
    print(f"Saving detection: {data}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080)
```

## Testing

### Test from Browser Console
```javascript
// Send current detection data
detector.sendCurrentDetectionData();

// Check detected items
detector.getDetectedItems();

// View current detection data
console.log(detector.detectionData);
```

### Test with Sample Data
```javascript
// Simulate detection
detector.detectionData = {
    excavator: { confidence: 85, detected: true, position: { x: 320, y: 240 } },
    pump: { confidence: 92, detected: true, position: { x: 450, y: 180 } }
};

// Send it
detector.sendCurrentDetectionData();
```

## Configuration

Detection transmission settings:
- **Transmission Interval:** 5 seconds (when detections present)
- **Retry Interval:** 30 seconds
- **Confidence Threshold:** 80% (configurable in settings)
- **Queue Max Size:** 100 items
- **Queue Expiry:** 1 hour

## Error Handling

- Failed WebSocket sends fall back to HTTP API
- Failed HTTP requests are queued for retry
- Queue stored in localStorage for persistence
- Console logging for debugging

## Security Considerations

- Validate all incoming data on backend
- Sanitize location coordinates
- Check confidence thresholds
- Rate limit API endpoints
- Use HTTPS/WSS in production
- Implement authentication for API endpoints


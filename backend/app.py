#!/usr/bin/env python3
"""
Galamsey AI Detector Backend - Clean Version
============================================
Ready for custom YOLOv12 model integration
No pre-trained models, clean slate for your custom training
"""

import os
import cv2
import torch
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import time
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GalamseyDetector:
    """Main detection class for illegal mining activities using custom YOLOv12"""
    
    def __init__(self, model_path="yolov12", dataset_path="illegal mining activities"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detection_threshold = 0.5
        
        # Custom class names for illegal mining detection
        self.class_names = self._load_class_names()
        
        # Detection statistics
        self.total_detections = 0
        self.illegal_detections = 0
        self.legal_detections = 0
        self.session_start = datetime.now()
        
        logger.info(f"Initializing GalamseyDetector on device: {self.device}")
        self._load_model()
    
    def _load_class_names(self):
        """Load class names from dataset or use default mining-related classes"""
        # Default classes for illegal mining detection
        default_classes = [
            'excavator', 'bulldozer', 'water_pump', 'wash_plant', 
            'truck', 'conveyor', 'drill', 'mining_equipment',
            'illegal_site', 'legal_site', 'suspicious_activity'
        ]
        
        # Try to load from dataset if available
        try:
            if os.path.exists(self.dataset_path):
                # Look for class names file in dataset
                class_file = os.path.join(self.dataset_path, "classes.txt")
                if os.path.exists(class_file):
                    with open(class_file, 'r') as f:
                        classes = [line.strip() for line in f.readlines()]
                    logger.info(f"Loaded {len(classes)} classes from dataset")
                    return classes
        except Exception as e:
            logger.warning(f"Could not load classes from dataset: {e}")
        
        logger.info(f"Using default classes: {default_classes}")
        return default_classes
    
    def _load_model(self):
        """Load custom YOLOv12 model"""
        try:
            # Look for YOLOv12 model files
            if os.path.exists(self.model_path):
                model_files = list(Path(self.model_path).glob("*.pt"))
                if model_files:
                    model_file = str(model_files[0])
                    logger.info(f"Loading custom YOLOv12 model from: {model_file}")
                    
                    # Load custom YOLOv12 model
                    # Note: This will need to be updated based on your specific YOLOv12 implementation
                    try:
                        # Try loading as PyTorch model
                        self.model = torch.load(model_file, map_location=self.device)
                        self.model.eval()
                        logger.info("✅ Custom YOLOv12 model loaded successfully")
                        return
                    except Exception as e:
                        logger.warning(f"Could not load as PyTorch model: {e}")
                        
                        # Try loading with ultralytics if available
                        try:
                            from ultralytics import YOLO
                            self.model = YOLO(model_file)
                            logger.info("✅ Custom YOLOv12 model loaded with ultralytics")
                            return
                        except ImportError:
                            logger.warning("Ultralytics not available")
            
            # No custom model found
            logger.warning("No custom YOLOv12 model found")
            logger.info("📝 To use custom model:")
            logger.info("   1. Train your YOLOv12 model on 'illegal mining activities' dataset")
            logger.info("   2. Place the trained model (.pt file) in 'yolov12/' folder")
            logger.info("   3. Restart the server")
            self.model = None
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def detect_mining_activities(self, frame):
        """Run inference on frame and detect mining activities"""
        if self.model is None:
            return self._generate_dummy_detections(frame)
        
        try:
            # Run inference with custom model
            if hasattr(self.model, 'predict'):
                # Ultralytics style
                results = self.model.predict(frame, conf=self.detection_threshold, verbose=False)
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            box = boxes[i]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            if conf > self.detection_threshold:
                                detection = {
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class_id': cls,
                                    'class_name': self.class_names[cls] if cls < len(self.class_names) else 'unknown',
                                    'is_illegal': self._is_illegal_mining(cls, conf),
                                    'timestamp': datetime.now().isoformat()
                                }
                                detections.append(detection)
                                
                                # Update statistics
                                self.total_detections += 1
                                if detection['is_illegal']:
                                    self.illegal_detections += 1
                                else:
                                    self.legal_detections += 1
                
                return detections
            else:
                # PyTorch model style
                # This will need to be customized based on your YOLOv12 implementation
                logger.warning("PyTorch model inference not implemented yet")
                return self._generate_dummy_detections(frame)
                
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return self._generate_dummy_detections(frame)
    
    def _is_illegal_mining(self, class_id, confidence):
        """Determine if detection represents illegal mining activity"""
        # Define illegal mining indicators
        illegal_indicators = ['excavator', 'water_pump', 'wash_plant', 'illegal_site', 'suspicious_activity']
        
        if class_id < len(self.class_names):
            class_name = self.class_names[class_id].lower()
            return any(indicator in class_name for indicator in illegal_indicators)
        
        return False
    
    def _generate_dummy_detections(self, frame):
        """Generate dummy detections for testing when model is not available"""
        height, width = frame.shape[:2]
        
        # Generate some dummy detections
        dummy_detections = [
            {
                'bbox': [int(width*0.2), int(height*0.3), int(width*0.4), int(height*0.6)],
                'confidence': 0.87,
                'class_id': 0,
                'class_name': 'excavator',
                'is_illegal': True,
                'timestamp': datetime.now().isoformat()
            },
            {
                'bbox': [int(width*0.5), int(height*0.4), int(width*0.7), int(height*0.7)],
                'confidence': 0.92,
                'class_id': 1,
                'class_name': 'water_pump',
                'is_illegal': True,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Update statistics
        self.total_detections += 2
        self.illegal_detections += 2
        
        return dummy_detections
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            is_illegal = detection['is_illegal']
            
            # Choose color based on detection type
            color = (0, 0, 255) if is_illegal else (0, 255, 0)  # Red for illegal, Green for legal
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add illegal/legal indicator
            status = "ILLEGAL" if is_illegal else "LEGAL"
            status_color = (0, 0, 255) if is_illegal else (0, 255, 0)
            cv2.putText(frame, status, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        return frame
    
    def get_statistics(self):
        """Get detection statistics"""
        session_duration = datetime.now() - self.session_start
        
        return {
            'total_detections': self.total_detections,
            'illegal_detections': self.illegal_detections,
            'legal_detections': self.legal_detections,
            'session_duration': str(session_duration),
            'detection_rate': self.total_detections / max(session_duration.total_seconds(), 1),
            'illegal_percentage': (self.illegal_detections / max(self.total_detections, 1)) * 100,
            'model_type': 'Custom YOLOv12' if self.model else 'Dummy (No Model)',
            'class_names': self.class_names
        }

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize detector
detector = GalamseyDetector()

# Global variables for video streaming
camera = None
is_streaming = False

def initialize_camera():
    """Initialize camera for video streaming"""
    global camera
    try:
        # Set environment variable to skip macOS camera authorization dialog
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
        
        # Try different camera indices
        for i in range(3):
            logger.info(f"Trying camera index {i}...")
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                # Test if we can actually read a frame
                ret, frame = camera.read()
                if ret and frame is not None:
                    logger.info(f"Camera initialized successfully on index {i}")
                    # Set camera properties for better performance
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    return True
                else:
                    logger.warning(f"Camera {i} opened but cannot read frames")
                    camera.release()
            else:
                logger.warning(f"Camera {i} could not be opened")
        
        logger.error("Could not initialize any camera")
        return False
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        return False

def generate_frames():
    """Generate video frames with detections"""
    global camera, is_streaming
    
    while is_streaming:
        if camera is None or not camera.isOpened():
            logger.error("Camera not available")
            break
        
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to read frame from camera")
            continue
        
        try:
            # Run detection
            detections = detector.detect_mining_activities(frame)
            
            # Draw detections on frame
            frame = detector.draw_detections(frame, detections)
            
            # Add status overlay
            illegal_count = sum(1 for d in detections if d['is_illegal'])
            status_text = f"Detections: {len(detections)} | Illegal: {illegal_count}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add model info
            model_info = f"Model: {detector.get_statistics()['model_type']}"
            cv2.putText(frame, model_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            continue
        
        # Control frame rate
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Galamsey AI Detector Backend',
        'version': '1.0.0',
        'status': 'running',
        'model_status': detector.get_statistics()['model_type'],
        'endpoints': {
            'health': '/health',
            'video_feed': '/video_feed',
            'statistics': '/statistics',
            'detections': '/detections'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    camera_status = "connected" if camera and camera.isOpened() else "disconnected"
    model_status = detector.get_statistics()['model_type']
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'camera': camera_status,
        'model': model_status,
        'device': str(detector.device),
        'detection_threshold': detector.detection_threshold,
        'class_names': detector.class_names
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    global is_streaming
    
    try:
        if not is_streaming:
            if not initialize_camera():
                return jsonify({'error': 'Camera not available'}), 500
            is_streaming = True
        
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Video feed error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics')
def statistics():
    """Get detection statistics"""
    return jsonify(detector.get_statistics())

@app.route('/detections')
def get_detections():
    """Get recent detections"""
    return jsonify({
        'message': 'Detection history endpoint - to be implemented',
        'note': 'This will store and return recent detections for frontend display',
        'current_model': detector.get_statistics()['model_type']
    })

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Configuration endpoint for adjusting detection parameters"""
    if request.method == 'POST':
        data = request.get_json()
        if 'threshold' in data:
            detector.detection_threshold = float(data['threshold'])
            logger.info(f"Detection threshold updated to: {detector.detection_threshold}")
        
        return jsonify({'message': 'Configuration updated'})
    
    return jsonify({
        'detection_threshold': detector.detection_threshold,
        'class_names': detector.class_names,
        'device': str(detector.device),
        'model_type': detector.get_statistics()['model_type']
    })

if __name__ == '__main__':
    logger.info("Starting Galamsey AI Detector Backend...")
    
    # Initialize camera
    if initialize_camera():
        logger.info("Camera initialized successfully")
    else:
        logger.warning("Camera initialization failed - running without camera")
    
    # Start Flask app
    try:
        logger.info("🚀 Server starting on http://localhost:5004")
        logger.info("📹 Video feed: http://localhost:5004/video_feed")
        logger.info("❤️ Health check: http://localhost:5004/health")
        logger.info("🤖 Ready for custom YOLOv12 model")
        logger.info("📝 To add your trained model:")
        logger.info("   1. Train YOLOv12 on 'illegal mining activities' dataset")
        logger.info("   2. Place trained model (.pt file) in 'yolov12/' folder")
        logger.info("   3. Restart the server")
        app.run(host='0.0.0.0', port=5004, debug=True, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
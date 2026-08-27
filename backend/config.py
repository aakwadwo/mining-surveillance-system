# Galamsey AI Detector Backend Configuration
# ==========================================
# Configuration file for different deployment environments

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Model Configuration
    MODEL_PATH = "yolov12"
    DATASET_PATH = "illegal mining activities"
    DETECTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.3
    
    # Camera Configuration
    CAMERA_INDEX = 0  # Default camera index
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    
    # Detection Classes
    ILLEGAL_CLASSES = [
        'excavator', 'bulldozer', 'water_pump', 'wash_plant',
        'truck', 'conveyor', 'drill', 'mining_equipment',
        'illegal_site', 'suspicious_activity'
    ]
    
    LEGAL_CLASSES = [
        'legal_site', 'authorized_equipment', 'permitted_operation'
    ]

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    MODEL_PATH = "yolov12"
    CAMERA_INDEX = 0

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    MODEL_PATH = "/opt/galamsey/models/yolov12"
    CAMERA_INDEX = 0  # May need adjustment for Pi camera

class RaspberryPiConfig(Config):
    """Raspberry Pi specific configuration"""
    DEBUG = False
    MODEL_PATH = "/home/pi/galamsey/models/yolov12"
    CAMERA_INDEX = 0  # Pi camera
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 15  # Lower FPS for Pi performance
    
    # Pi-specific optimizations
    TORCH_THREADS = 2
    OPENCV_THREADS = 2

# Environment-based configuration
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'raspberry_pi':
        return RaspberryPiConfig()
    else:
        return DevelopmentConfig()

#!/usr/bin/env python3
"""
Simple test server to debug the Galamsey AI Detector backend
"""

import sys
import os

print("🔍 Testing imports...")

try:
    import cv2
    print("✅ OpenCV imported successfully")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch imported successfully - Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy imported successfully - Version: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")

try:
    from flask import Flask, jsonify
    print("✅ Flask imported successfully")
except Exception as e:
    print(f"❌ Flask import failed: {e}")

# Test camera
print("\n📹 Testing camera...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera is accessible")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera frame captured: {frame.shape}")
        else:
            print("❌ Could not read frame from camera")
        cap.release()
    else:
        print("❌ Camera is not accessible")
except Exception as e:
    print(f"❌ Camera test failed: {e}")

# Simple Flask server
print("\n🚀 Starting simple test server...")
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Galamsey AI Detector Test Server',
        'status': 'running',
        'imports': {
            'opencv': 'cv2' in sys.modules,
            'torch': 'torch' in sys.modules,
            'numpy': 'numpy' in sys.modules,
            'flask': 'flask' in sys.modules
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Test server is running'
    })

if __name__ == '__main__':
    print("🌐 Server will be available at: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

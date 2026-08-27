#!/usr/bin/env python3
"""
Camera Permission Test for macOS
================================
This script will help you grant camera permissions to Python/Terminal
"""

import cv2
import os

print("🎥 Camera Permission Test for macOS")
print("=" * 40)

# Set environment variable to handle macOS camera permissions
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

print("📋 Instructions:")
print("1. If prompted, grant camera access to Terminal/Python")
print("2. You may need to go to System Preferences > Security & Privacy > Camera")
print("3. Make sure Terminal (or your IDE) has camera access enabled")
print()

# Test camera access
print("🔍 Testing camera access...")

for i in range(3):
    print(f"Trying camera index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        print(f"✅ Camera {i} opened successfully")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Camera {i} can read frames: {frame.shape}")
            cap.release()
            print(f"🎉 SUCCESS! Camera {i} is working!")
            break
        else:
            print(f"❌ Camera {i} opened but cannot read frames")
    else:
        print(f"❌ Camera {i} could not be opened")
    
    cap.release()

print()
print("💡 If no camera worked:")
print("1. Check System Preferences > Security & Privacy > Camera")
print("2. Make sure Terminal has camera access")
print("3. Try running this script from your IDE instead of Terminal")
print("4. Restart Terminal/IDE after granting permissions")

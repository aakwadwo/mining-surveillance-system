#!/bin/bash

# Galamsey AI Detector Backend Setup Script
# =========================================
# This script sets up the backend environment for illegal mining detection

echo "🚁 Setting up Galamsey AI Detector Backend..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ required. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if CUDA is available (optional)
echo "🔍 Checking for CUDA support..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p data

# Set permissions
chmod +x app.py

echo "✅ Backend setup complete!"
echo ""
echo "🚀 To start the backend server:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🌐 The server will be available at:"
echo "   http://localhost:5000"
echo "   http://localhost:5000/health (health check)"
echo "   http://localhost:5000/video_feed (video stream)"
echo ""
echo "📱 For Raspberry Pi deployment:"
echo "   export FLASK_ENV=raspberry_pi"
echo "   python app.py"

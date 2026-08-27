#!/bin/bash
# Superseded. Environment setup now happens at the project root.
#
# The original version of this script created backend/venv with the
# system Python 3.9 and announced port 5000. Both are wrong now:
# the project uses a root .venv on Python 3.12, and port 5004.
set -e
cd "$(dirname "$0")/.."
echo "Setting up the Galamsey AI Detector environment..."
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
echo
echo "Done. Start the backend with:"
echo "    ./.venv/bin/python backend/app.py"
echo "Then open http://127.0.0.1:5004/health"

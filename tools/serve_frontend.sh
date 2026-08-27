#!/bin/bash
# Serve the frontend over HTTP.
#
# WHY NOT JUST DOUBLE-CLICK index.html?
# -------------------------------------
# Opening the file gives you a file:// URL, which browsers treat as an
# INSECURE CONTEXT. In an insecure context the browser disables:
#
#   navigator.mediaDevices.getUserMedia  -> the webcam button does nothing
#   navigator.geolocation                -> map location tracking is blocked
#   BroadcastChannel across tabs         -> the map pop-out cannot sync
#
# and fetch() to the backend is blocked by CORS, because a file:// page has
# an opaque "null" origin that no server can whitelist.
#
# http://localhost IS a secure context, even without HTTPS - browsers make
# that exception specifically so local development works. So we serve it.
#
# Port 5501 matches frontend/.vscode/settings.json (Live Server), and is in
# the backend's CORS allow-list in backend/config.py.
set -e
PORT="${1:-5501}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/frontend"
echo "Serving $DIR"
echo "  ->  http://localhost:$PORT"
echo "Backend must be running separately:  ./.venv/bin/python backend/app.py"
echo
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR"

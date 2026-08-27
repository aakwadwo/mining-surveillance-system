#!/bin/bash
# Start backend AND frontend together for development.
# Ctrl+C stops both.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run:  python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

cleanup() { echo; echo "stopping..."; kill 0 2>/dev/null; }
trap cleanup EXIT INT TERM

./.venv/bin/python backend/app.py &
python3 -m http.server 5501 --bind 127.0.0.1 --directory frontend >/dev/null 2>&1 &

sleep 2
echo
echo "==================================================="
echo "  frontend : http://localhost:5501"
echo "  backend  : http://127.0.0.1:5004/health"
echo "  Ctrl+C to stop both"
echo "==================================================="
wait

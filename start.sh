#!/bin/bash
echo ""
echo "====================================================="
echo "  TrafficLens - Starting Backend"
echo "====================================================="
echo ""
cd "$(dirname "$0")/backend"
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting server on http://localhost:8000"
echo "Open frontend/index.html in your browser"
echo "Press Ctrl+C to stop"
echo ""
uvicorn main:app --reload --port 8000

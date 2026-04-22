#!/bin/bash
set -e

# Project paths
PROJECT_ROOT=$(pwd)
BACKEND_DIR="$PROJECT_ROOT/webapp/backend"
FRONTEND_DIR="$PROJECT_ROOT/webapp/frontend"
VENV_DIR="$PROJECT_ROOT/webapp_venv"

echo "=============================================="
echo "🧠 Starting NeuroScan AI Webapp..."
echo "=============================================="

# 1. Setup python virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "📥 Installing/verifying backend dependencies..."
pip install --upgrade pip -q
pip install -r "$BACKEND_DIR/requirements.txt" -q

# 2. Start the Backend in the background
echo "🚀 Starting FastAPI Backend on http://localhost:8000..."
cd "$BACKEND_DIR"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment to let the backend start up
sleep 3

# 3. Start the Frontend
echo "🌐 Starting Frontend Server on http://localhost:8080..."
echo "👉 OPEN YOUR BROWSER TO: http://localhost:8080"
echo "Press Ctrl+C to stop both servers."
echo "----------------------------------------------"

cd "$FRONTEND_DIR"

# Cleanup function to kill background backend when script exits (Ctrl+C)
cleanup() {
    echo ""
    echo "🛑 Stopping NeuroScan AI Webapp servers..."
    kill $BACKEND_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM EXIT

# Start the python HTTP server in the foreground
python3 -m http.server 8080

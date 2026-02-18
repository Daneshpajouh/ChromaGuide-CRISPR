#!/bin/bash
"""
ChromaGuide API - Quick Start Script

Usage:
    ./start_api.sh [dev|prod] [port]

Examples:
    ./start_api.sh dev 8000
    ./start_api.sh prod 8000
"""

# Default settings
MODE=${1:-dev}
PORT=${2:-8000}
WORKERS=${3:-4}

echo "======================================================================"
echo "ChromaGuide REST API - Starting"
echo "======================================================================"
echo "Mode:    $MODE"
echo "Port:    $PORT"
echo "Workers: $WORKERS"
echo "======================================================================"

# Check if running
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "✗ Port $PORT already in use. Choose a different port:"
    echo "  ./start_api.sh $MODE 8001"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements_api.txt
else
    source venv/bin/activate
fi

# Start server
if [ "$MODE" = "dev" ]; then
    echo ""
    echo "Starting development server with auto-reload..."
    echo "Interactive docs: http://localhost:$PORT/docs"
    echo ""
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port $PORT
else
    echo ""
    echo "Starting production server..."
    echo "Interactive docs: http://localhost:$PORT/docs"
    echo ""
    gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:$PORT \
        --access-logfile - \
        --error-logfile - \
        src.api.main:app
fi

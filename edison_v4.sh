#!/bin/bash
# Edison Research Hub v8.0 - Recovery & Launch Script
# Optimized for Mac Studio M3 Ultra (96GB RAM)

BANNER="
  _____ ____ ___ ____   ___  _   _    __     ___
 | ____|  _ \_ _/ ___| / _ \| \ | |   \ \   / / |
 |  _| | | | | |\___ \| | | |  \| |    \ \ / /| |
 | |___| |_| | | ___) | |_| | |\  |     \ V / | |
 |_____|____/___|____/ \___/|_| \_|      \_/  |_|

           [ SOTA RESEARCH HUB : VERSION 8.0 ]
"

# 1. Initialization
clear
echo "$BANNER"
echo "🛠️  [SYSTEM] Initializing Edison v8.0 Recovery Protocol..."

# 2. Port Cleanup
echo "🧹 [CLEANUP] Purging stale sessions on ports 8000 (API) and 5173 (UI)..."
lsof -ti:8000,5173 | xargs kill -9 2>/dev/null
sleep 1

# 3. Environment Check
echo "🔍 [CHECK] Verifying platform dependencies..."
if ! command -v uvicorn &> /dev/null; then
    echo "❌ [ERROR] uvicorn not found. Please install requirements.txt"
    exit 1
fi

# 4. Starting Backend
echo "📡 [BACKEND] Launching Global Knowledge Bridge (FastAPI)..."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
nohup uvicorn research_platform.gui.server:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend running (PID: $BACKEND_PID)"

# 5. Starting Frontend
echo "💻 [FRONTEND] Launching Specialist Research Lab UI (Vite)..."
cd research_platform/gui/client
nohup npm run dev > ../../../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd "../../.."
echo "✅ Frontend running (PID: $FRONTEND_PID)"

# 6. Final Status
echo ""
echo "=================================================="
echo "🚀 EDISON v8.0 IS NOW LIVE"
echo "--------------------------------------------------"
echo "🌐 RESEARCH HUB:  http://localhost:5173"
echo "📊 API TELEMETRY: http://localhost:8000/docs"
echo "📂 PROCESS LOGS:  ./logs/"
echo "=================================================="
echo ""
echo "📡 [SYSTEM] Monitoring heartbeats. Press Ctrl+C to stop all services (if running in foreground)."
echo "Note: This script launched processes in the background for stability."

# Optional: Wait for user input to keep script alive and then cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID; echo -e '\n🛑 Edison Lab is now Offline.'; exit" SIGINT
wait

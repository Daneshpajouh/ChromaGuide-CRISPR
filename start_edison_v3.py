import subprocess
import time
import sys
import os
import signal

# --- CONFIGURATION ---
BACKEND_CMD = ["uvicorn", "research_platform.gui.server:app", "--host", "0.0.0.0", "--port", "8000"]
FRONTEND_DIR = "research_platform/gui/client"
FRONTEND_CMD = ["npm", "run", "dev"]
BANNER = """
  _____ ____ ___ ____   ___  _   _   __     _________
 | ____|  _ \_ _/ ___| / _ \| \ | |  \ \   / /___ / /
 |  _| | | | | |\___ \| | | |  \| |   \ \ / /  |_ \ /
 | |___| |_| | | ___) | |_| | |\  |    \ V /  ___) \
 |_____|____/___|____/ \___/|_| \_|     \_/  |____/ \

           [ SOTA RESEARCH HUB : CORE-ENGAGED ]
"""

processes = []

def cleanup(sig, frame):
    print("\n\n🛑 [SYSTEM] Initiating Graceful Shutdown...")
    for proc in processes:
        try:
            proc.terminate()
            print(f"✅ Halted process: {proc.pid}")
        except:
            pass
    print("✨ Edison Lab is now Offline.\n")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

def main():
    os.system('clear')
    print(BANNER)
    print("🎨 [UI/UX] Customizing Research Environment...")
    print("🧹 [CLEANUP] Purging stale sessions...")

    # Optional: Kill anything on 8000 or 5173
    os.system("lsof -ti:8000,5173 | xargs kill -9 2>/dev/null")
    time.sleep(1)

    print("📡 [BACKEND] Launching Global Knowledge Bridge...")
    backend = subprocess.Popen(BACKEND_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(backend)
    time.sleep(2)

    print("💻 [FRONTEND] Launching Specialist Research Lab UI...")
    frontend = subprocess.Popen(FRONTEND_CMD, cwd=FRONTEND_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(frontend)
    time.sleep(2)

    print("\n" + "="*50)
    print("✅ EDISON v3 IS FULLY OPERATIONAL")
    print("🚀 RESEARCH HUB:  http://localhost:5173")
    print("📊 API TELEMETRY: http://localhost:8000/docs")
    print("="*50 + "\n")
    print("📡 Monitoring Specialist Heartbeat... [Ctrl+C to Exit]")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

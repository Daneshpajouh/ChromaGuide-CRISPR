import subprocess
import time
import os
import sys

def start_edison():
    print("🎨 Starting Edison Command Center v2...")

    # 1. Kill old ones
    print("🧹 Cleaning up old sessions...")
    subprocess.run("pkill -9 -f 'server.py'", shell=True)
    subprocess.run("pkill -9 -f 'vite'", shell=True)
    time.sleep(2)

    # 2. Start Backend
    print("📡 Launching Global Knowledge Bridge (Backend)...")
    backend = subprocess.Popen(
        ["python3", "research_platform/gui/server.py"],
        stdout=open("research_platform/gui/logs/server.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=os.getcwd()
    )

    # 3. Start Frontend
    print("💻 Launching Immersive Lab UI (Frontend)...")
    frontend = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "--port", "3000"],
        stdout=open("research_platform/gui/logs/client.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=os.path.join(os.getcwd(), "research_platform/gui/client")
    )

    print("\n" + "="*50)
    print("✅ EDISON v2 IS CORE-ENGAGED!")
    print("🚀 DASHBOARD: http://localhost:3000")
    print("="*50)
    print("\nPress Ctrl+C to shut down all systems.")

    try:
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                print("❌ Backend crashed. Restarting...")
                break
            if frontend.poll() is not None:
                print("❌ Frontend crashed. Restarting...")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend.terminate()
        frontend.terminate()

if __name__ == "__main__":
    start_edison()

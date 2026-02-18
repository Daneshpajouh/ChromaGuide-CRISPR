import sqlite3
import sys
import os
import psutil
import time
import asyncio
from typing import Dict, List
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.master_stack import ResearchMaster
from research_platform.local_file_handler import LocalFileHandler

app = FastAPI(title="Edison Command Center")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class PlatformState:
    def __init__(self):
        self.is_busy = False
        self.current_task = "Idle"
        self.progress = 0
        self.last_report = None
        self.master = ResearchMaster()
        self.interrupt_signal = asyncio.Event()
        self.events = [] # Event buffer for client polling
        self.logs = [] # To avoid AttributeError if accessed before set

state = PlatformState()

class InvestigationRequest(BaseModel):
    topic: str
    tier: str = "deep_apex"
    context: str = "" # Optional previous research context
    file_paths: List[str] = [] # Local files to analyze

@app.get("/api/hardware")
async def get_hardware_stats():
    """Get M3 Ultra telemetry."""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage("/").free / (1024**3), 1)
    }

@app.get("/api/status")
async def get_status():
    return {
        "is_busy": state.is_busy,
        "current_task": state.current_task,
        "progress": state.progress,
        "last_report": state.last_report,
        "logs": state.logs[-50:] if hasattr(state, 'logs') else []
    }

@app.get("/api/events")
async def get_events():
    """Returns and clears the event buffer."""
    evs = list(state.events)
    state.events = []
    return evs

@app.get("/api/graph")
async def get_graph():
    """Returns all entities and relationships for visualization."""
    entities = state.master.graph.get_all_entities()
    rels = state.master.graph.get_all_relations()

    return {
        "nodes": [{"id": e[0], "label": e[1], "type": e[2]} for e in entities],
        "links": [{"source": r[0], "target": r[1], "label": r[2]} for r in rels]
    }

@app.get("/api/hypotheses")
async def get_hypotheses():
    """Returns structured hypothesis evolution data."""
    hyps = state.master.hypothesis_engine.hypotheses
    return {hid: {
        "content": h.content,
        "confidence": h.confidence_score,
        "status": h.status,
        "evidence_count": len(h.evidence),
        "created_at": h.created_at
    } for hid, h in hyps.items()}

@app.delete("/api/logs")
async def clear_logs():
    state.logs = []
    return {"message": "Logs cleared"}

async def run_investigation_driver(topic: str, tier: str, context_str: str = "", file_paths: List[str] = []):
    state.is_busy = True
    state.progress = 0
    state.logs = [] # Clear logs at the start of a new investigation
    state.logs.append(f"🌌 Launching {tier.replace('_', ' ').upper()} Directive: {topic}")

    # Process Attached Files
    if file_paths:
        state.logs.append(f"📂 Processing {len(file_paths)} local attachments...")
        try:
            handler = LocalFileHandler()
            file_context = handler.process_files(file_paths)
            context_str += f"\n\n{file_context}"
            state.logs.append("✅ Local files ingested into Research Context.")
        except Exception as e:
            state.logs.append(f"⚠️ File Processing Warning: {e}")
    try:
        # Execute Dynamic Master Investigation
        # The master.investigate method is now responsible for updating current_task, progress, and last_report
        # It also handles the detailed logging via the log_callback.
        # Custom log callback to also buffer events
        def log_handler(m):
            state.logs.append(m)
            if m.startswith("EVENT:"):
                state.events.append(m.replace("EVENT:", ""))

        report_filename = await state.master.investigate(
            topic,
            tier=tier,
            context_str=context_str,
            log_callback=log_handler,
            progress_callback=lambda p: setattr(state, 'progress', p),
            task_callback=lambda t: setattr(state, 'current_task', t)
        )
        state.last_report = report_filename
        state.logs.append(f"🏆 Strategic Synthesis Successful.")
    except asyncio.CancelledError:
        state.logs.append("🛑 Investigation Halted by User.")
        state.current_task = "Halted"
    except Exception as e:
        state.logs.append(f"❌ Error: {str(e)}")
        state.current_task = "Error"
    finally:
        state.is_busy = False
        state.progress = 100

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file uploads."""
    upload_dir = "research_platform/uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename, "path": os.path.abspath(file_path)}

@app.post("/api/stop")
async def stop_investigation():
    if not state.is_busy:
        return {"message": "No active investigation."}
    # We'll use a simple background task flag check or similar if needed,
    # but for now, we'll mark the state as not busy.
    state.is_busy = False
    state.logs.append("🛑 Manual Stop Signal Received.")
    return {"message": "Stop signal sent."}

@app.post("/api/investigate")
async def start_investigation(req: InvestigationRequest, background_tasks: BackgroundTasks):
    if state.is_busy:
        raise HTTPException(status_code=400, detail="Platform is currently busy.")

    background_tasks.add_task(run_investigation_driver, req.topic, req.tier, req.context, req.file_paths)
    return {"message": "Investigation started."}

@app.delete("/api/report/{filename}")
async def delete_report(filename: str):
    path = os.path.join("research_platform/output", filename)
    print(f"🗑️ Deleting report: {path}")
    if not os.path.exists(path):
        print(f"⚠️ Report not found: {path}")
        raise HTTPException(status_code=404, detail="Report not found")
    os.remove(path)
    print(f"✅ Deleted: {path}")
    return {"message": "Report deleted"}

class ReportUpdate(BaseModel):
    content: str

@app.put("/api/report/{filename}")
async def update_report(filename: str, req: ReportUpdate):
    path = os.path.join("research_platform/output", filename)
    print(f"💾 Updating report: {path}")
    if not os.path.exists(path):
        print(f"⚠️ Report not found: {path}")
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "w") as f:
        f.write(req.content)
    print(f"✅ Updated: {path}")
    return {"message": "Report updated"}

@app.get("/api/consensus")
async def get_consensus():
    """Returns the latest consensus metrics from the ensemble."""
    return state.master.ensemble.get_metrics()

@app.get("/api/reports")
async def list_reports():
    output_dir = "research_platform/output"
    if not os.path.exists(output_dir):
        return []
    files = [f for f in os.listdir(output_dir) if f.endswith(".md")]
    return sorted(files, reverse=True)

@app.get("/api/report/{filename}")
async def get_report(filename: str):
    path = os.path.join("research_platform/output", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r") as f:
        return {"content": f.read()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

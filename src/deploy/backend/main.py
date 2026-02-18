from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.deploy.backend.schemas import InferenceRequest, InferenceResponse, RiskScore
from src.deploy.backend.inference import ModelService

app = FastAPI(
    title="CRISPRO-Mamba Clinical Interface",
    description="Real-time off-target risk prediction API using Mamba-S6 architecture.",
    version="1.0.0"
)

# CORS - Allow React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Service
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../checkpoints/best_model.pth")
model_service = ModelService(model_path=MODEL_PATH)

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "CRISPRO-Mamba-S6",
        "device": str(model_service.device),
        "model_loaded": model_service.model is not None
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict_risk(request: InferenceRequest):
    if not model_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Perform inference
        risk_score = model_service.predict(request.sequence)

        # In a real scenario, we'd map this to specific off-target sites or create a risk map.
        # For now, we return the global efficiency/risk score.

        return {
            "target_sequence": request.sequence,
            "off_target_risk": risk_score,
            "risk_map": [
                RiskScore(position=0, score=risk_score, context="Global Risk")
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

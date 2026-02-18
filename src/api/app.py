from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
from src.model.crispro import CRISPROModel
# We need to import the GenomeLoader and Tokenizer logic here too,
# or wrap them in a service.

app = FastAPI(title="CRISPRO-MAMBA-X Clinical Dashboard", version="1.0.0")

class PredictionRequest(BaseModel):
    sequence: str
    pam_index: int
    genome_ver: str = "hg38"
    cell_type: Optional[str] = None # For future epigenetic lookup

class PredictionResponse(BaseModel):
    guide_sequence: str
    efficiency_score: float # 0-100
    active_probability: float # 0-1
    safety_tier: str # High/Medium/Low
    off_target_risk: str

# Global Model Placeholder
model = None
device = "mps" if torch.backends.mps.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    global model
    print(f"Loading Model on {device}...")
    # Load logic will go here once training finishes
    # model = CRISPROModel(...)
    # model.load_state_dict(...)
    print("Model Loaded (Placeholder).")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict CRISPR Efficiency and Safety for a given guide sequence.
    """
    # 1. Validation
    if len(request.sequence) < 20:
        raise HTTPException(status_code=400, detail="Sequence too short")

    # 2. Inference (Mock for now)
    # In real implementation:
    #   features = encode(request.sequence)
    #   cls, reg = model(features)

    mock_efficiency = 85.5
    mock_prob = 0.92

    return PredictionResponse(
        guide_sequence=request.sequence,
        efficiency_score=mock_efficiency,
        active_probability=mock_prob,
        safety_tier="High Efficiency",
        off_target_risk="Low"
    )

@app.get("/health")
async def health():
    return {"status": "online", "model_device": device}

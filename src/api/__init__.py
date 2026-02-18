"""ChromaGuide REST API Package

REST API for CRISPR sgRNA design prediction with:
- On-target efficiency prediction
- Off-target risk assessment
- Integrated design scoring
- Conformal prediction intervals
- Batch processing support

Main modules:
- main.py: FastAPI application and endpoints
- schemas.py: Pydantic request/response models
- inference.py: Model loading and prediction logic
"""

from .inference import (
    ChromaGuideInferenceEngine,
    ConformalPredictor,
    get_inference_engine,
    reinitialize_engine,
)

__all__ = [
    'ChromaGuideInferenceEngine',
    'ConformalPredictor',
    'get_inference_engine',
    'reinitialize_engine',
]

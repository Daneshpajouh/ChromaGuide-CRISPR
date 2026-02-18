"""
CHROMAGUIDE QUICK REFERENCE CARD

One-page summary of factory and API usage.
"""

# ==============================================================================
# MODEL FACTORY - Quick Start
# ==============================================================================

from src.model import create_model, load_model, ModelFactory, list_available_models

# 1. Create a model
model = create_model('chromaguide', d_model=256, use_epigenomics=True)

# 2. Load from checkpoint
model = load_model('checkpoints/best.pt', 'chromaguide')

# 3. Save model
from src.model import save_model
save_model(model, 'checkpoints/model.pt', model_name='chromaguide')

# 4. Use factory directly
factory = ModelFactory(device='cuda')
model = factory.create('crispro', n_layers=4)

# 5. List available models
models = list_available_models()
for name, desc in models.items():
    print(f"{name}: {desc}")

# 6. Batch create
models = factory.create_ensemble(['crispro', 'chromaguide', 'deepmens'])

# 7. Custom config
config = {'d_model': 512, 'n_layers': 8}
model = create_model('crispro', config=config)

# 8. Parameter utilities
from src.model import count_parameters, freeze_backbone, unfreeze_backbone
num_params = count_parameters(model)
freeze_backbone(model, freeze_all_but=['head'])


# ==============================================================================
# REST API - Quick Start
# ==============================================================================

# Start server (development)
# $ uvicorn src.api.main:app --reload --port 8000
# $ open http://localhost:8000/docs

# Start server (production)
# $ gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app

# Start with Docker
# $ docker build -f Dockerfile.api -t chromaguide-api .
# $ docker run -p 8000:8000 chromaguide-api

# Start with Docker Compose
# $ docker-compose up chromaguide-api


# ==============================================================================
# API - Python Client
# ==============================================================================

from examples.chromaguide_client import ChromaGuideClient

client = ChromaGuideClient(api_url="http://localhost:8000")

# Single prediction
result = client.predict_single("ACGTACGTACGTACGTACGTACG")
print(f"Design Score: {result.design_score}")
print(f"Efficiency: {result.efficiency_score}")

# Batch prediction
sequences = ["ACGT...", "TGCA...", "AAAA..."]
batch_result = client.predict_batch(sequences)
print(f"Processed {batch_result.num_guides} guides in {batch_result.processing_time_sec:.2f}s")

# Design score calculation
score = client.calculate_design_score(
    efficiency=0.8,
    off_target_risk=0.1,
    specificity=0.7
)
print(f"Design Score: {score['design_score']}")

# Health check
health = client.health_check()
print(f"Model Loaded: {health['model_loaded']}")
print(f"GPU Available: {health['gpu_available']}")


# ==============================================================================
# API - REST Endpoints (curl)
# ==============================================================================

# Health check
# $ curl http://localhost:8000/health

# Single prediction
# $ curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"sequence":"ACGTACGTACGTACGTACGTACG"}'

# Batch prediction
# $ curl -X POST http://localhost:8000/predict/batch \
#   -H "Content-Type: application/json" \
#   -d '{
#     "guides": [
#       {"sequence":"ACGT..."},
#       {"sequence":"TGCA..."}
#     ]
#   }'

# List models
# $ curl -X POST http://localhost:8000/models/info -H "Content-Type: application/json" -d '{}'

# Load specific model
# $ curl -X POST "http://localhost:8000/models/load?model_name=crispro"


# ==============================================================================
# API - Response Format
# ==============================================================================

# Single prediction response:
{
    "guide_sequence": "ACGTACGTACGTACGTACGTACG",
    "efficiency_score": 0.78,
    "efficiency_lower": 0.65,
    "efficiency_upper": 0.91,
    "off_target_risk": 0.12,
    "off_target_lower": 0.08,
    "off_target_upper": 0.16,
    "design_score": 0.83,
    "safety_tier": "low_risk",
    "activity_probability": 0.78,
    "specificity_score": 0.65
}


# ==============================================================================
# Available Models
# ==============================================================================

"""
CORE MODELS:
  chromaguide           - Multi-modal (sequence + epigenomics) with Beta regression
  crispro               - Mamba-2 for on-target efficiency
  crispro_mamba_x      - CRISPRO + quantum tunneling + topological DA
  dnabert_mamba        - DNABERT foundation + Mamba adapter

ENSEMBLE/ALTERNATIVE:
  deepmens              - Multi-scale CNN (sequence + shape + position)
  deepmens_ensemble     - 5-model voting ensemble
  mamba_deepmens       - Mamba + DeepMENS hybrid

SPECIALIZED:
  off_target            - CNN-based off-target risk
  ot_rag                - Retrieval-augmented off-target prediction
  conformal             - Conformal prediction wrapper
  
GENERATIVE:
  rnagenesis_vae        - VAE for synthetic guide generation
  rnagenesis_diffusion  - Diffusion model for generation
"""


# ==============================================================================
# Configuration Files
# ==============================================================================

# requirements_api.txt
# Install all API dependencies:
# $ pip install -r requirements_api.txt

# start_api.sh
# Quick server startup:
# $ chmod +x start_api.sh
# $ ./start_api.sh dev 8000        # Development
# $ ./start_api.sh prod 8000       # Production

# Dockerfile.api
# Build Docker image:
# $ docker build -f Dockerfile.api -t chromaguide-api .

# docker-compose.yml
# Start full stack (API + Redis + PostgreSQL + Monitoring):
# $ docker-compose up
# API: http://localhost:8000
# Grafana: http://localhost:3000


# ==============================================================================
# Testing
# ==============================================================================

# Run full API test suite:
# $ python examples/test_api.py

# Or start server and test with Python:
# $ python
# >>> from examples.chromaguide_client import predict_single
# >>> result = predict_single("ACGTACGTACGTACGTACGTACG")
# >>> print(result)


# ==============================================================================
# Performance Tips
# ==============================================================================

# 1. Use batch predictions for multiple guides
#    Batch: 10-100x faster than individual requests

# 2. Cache model predictions
#    API automatically caches identical sequences

# 3. Use GPU for inference
#    Set device='cuda' in factory or ModelFactory

# 4. Warm up model with health check
#    First request includes model initialization time

# 5. Reduce batch size for large memory constraints
#    Default batch size: 32

# 6. Use conformal prediction intervals (~10% overhead)
#    Provides statistical guarantees on predictions


# ==============================================================================
# Troubleshooting
# ==============================================================================

# Model not loading?
# $ uvicorn src.api.main:app --log-level debug

# Out of memory?
# $ export CUDA_VISIBLE_DEVICES=""  # Use CPU instead
# $ docker run -m 8gb chromaguide-api  # Limit Docker memory

# Port already in use?
# $ lsof -i :8000
# $ kill -9 <PID>
# $ ./start_api.sh dev 8001  # Use different port

# Invalid sequence error?
# - Only ACGT and N characters are supported
# - Sequences are auto-converted to uppercase
# - Strip whitespace: seq.strip()


# ==============================================================================
# Key Files
# ==============================================================================

Production Code:
  src/model/registry.py              - Model registry
  src/model/factory.py               - Factory implementation
  src/model/model_zoo.py             - Model registration
  src/model/utils.py                 - Convenience functions
  src/api/schemas.py                 - Pydantic models
  src/api/inference.py               - Prediction engine
  src/api/main.py                    - FastAPI app

Documentation:
  CHROMAGUIDE_FACTORY_GUIDE.md       - Factory complete guide
  API_COMPLETE_GUIDE.md              - API documentation
  IMPLEMENTATION_SUMMARY_P1_P2.md   - Implementation summary

Examples:
  examples/train_with_factory.py     - Training example
  examples/inference_with_factory.py - Inference example
  examples/test_api.py               - API test suite
  examples/chromaguide_client.py     - Python client

Configuration:
  requirements_api.txt               - API dependencies
  start_api.sh                       - Startup script
  Dockerfile.api                     - Docker image
  docker-compose.yml                 - Full stack

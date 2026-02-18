"""
ChromaGuide REST API - Complete Documentation

This document provides comprehensive guide to using the ChromaGuide REST API
for CRISPR sgRNA design prediction.

Table of Contents:
  1. Getting Started
  2. API Endpoints
  3. Request/Response Examples
  4. Error Handling
  5. Deployment
  6. Performance Optimization
  7. Troubleshooting
"""

# ==============================================================================
# 1. GETTING STARTED
# ==============================================================================

"""
INSTALLATION

pip install fastapi uvicorn torch numpy

STARTING THE SERVER

Option 1: Development server with auto-reload
    cd /path/to/chromaguide
    uvicorn src.api.main:app --reload --port 8000

Option 2: Production server (requires gunicorn)
    pip install gunicorn
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app --bind 0.0.0.0:8000

Option 3: With Docker
    docker build -t chromaguide-api .
    docker run -p 8000:8000 chromaguide-api

Option 4: With model checkpoint
    export CHROMAGUIDE_MODEL=chromaguide
    export CHROMAGUIDE_CHECKPOINT=/path/to/checkpoint.pt
    uvicorn src.api.main:app --port 8000

ACCESSING THE API

Interactive documentation (Swagger UI):
    http://localhost:8000/docs

Alternative documentation (ReDoc):
    http://localhost:8000/redoc

Health check:
    curl http://localhost:8000/health

API root:
    curl http://localhost:8000/
"""

# ==============================================================================
# 2. API ENDPOINTS
# ==============================================================================

"""
ENDPOINT SUMMARY

Root Endpoints:
  GET  /              - API information
  GET  /health        - Health check

Prediction Endpoints:
  POST /predict       - Single sgRNA prediction
  POST /predict/batch - Batch predictions

Model Management:
  POST /models/info   - Get available models
  POST /models/load   - Load specific model

Utilities:
  POST /design-score  - Calculate design score

DETAILED ENDPOINT DOCUMENTATION

1. GET /health
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Check API and model health status.
   
   Response:
   {
       "status": "healthy",
       "version": "1.0.0",
       "model_loaded": true,
       "gpu_available": true,
       "memory_usage_mb": 2048.5,
       "uptime_seconds": 3600.0,
       "details": {
           "num_predictions": 42,
           "timestamp": "2026-02-16T10:00:00"
       }
   }


2. POST /predict
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Single sgRNA prediction.
   
   Request:
   {
       "sequence": "ACGTACGTACGTACGTACGTACG",
       "cas_type": "cas9",
       "off_target_threshold": 0.1,
       "include_uncertainty": true,
       "include_explanations": false,
       "genome_version": "hg38"
   }
   
   Response:
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


3. POST /predict/batch
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Batch predictions for multiple guides.
   
   Request:
   {
       "guides": [
           {"sequence": "ACGTACGTACGTACGTACGTACG"},
           {"sequence": "TGCATGCATGCATGCATGCATGCA"},
           {"sequence": "AAAATTTTCCCCGGGGACGTACGT"}
       ],
       "return_all": false
   }
   
   Response:
   {
       "predictions": [...],
       "num_guides": 3,
       "num_high_quality": 2,
       "top_guides": [...top 10...],
       "processing_time_sec": 0.25
   }


4. POST /models/info
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Get available models.
   
   Request:
   {
       "model_name": null  # Optional specific model
   }
   
   Response:
   {
       "available_models": ["chromaguide", "crispro", "deepmens", ...],
       "models_info": null  # Or detailed info if model_name specified
   }


5. POST /models/load
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Load a specific model.
   
   Query Parameters:
       model_name (required): Model identifier
       checkpoint_path (optional): Path to checkpoint
   
   Response:
   {
       "status": "success",
       "message": "Loaded model 'crispro'",
       "model_info": {
           "model_loaded": true,
           "num_parameters": 45000000,
           "load_time_sec": 2.5
       }
   }


6. POST /design-score
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Calculate design score from components.
   
   Request:
   {
       "efficiency_score": 0.78,
       "off_target_risk": 0.12,
       "specificity_score": 0.65,
       "efficiency_weight": 0.5,
       "safety_weight": 0.5
   }
   
   Response:
   {
       "design_score": 0.83,
       "components": {
           "efficiency": 0.78,
           "safety": 0.88,
           "specificity": 0.65
       }
   }
"""

# ==============================================================================
# 3. REQUEST/RESPONSE EXAMPLES
# ==============================================================================

"""
EXAMPLE 1: Simple Single Prediction

curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "sequence": "ACGTACGTACGTACGTACGTACG",
    "cas_type": "cas9",
    "include_uncertainty": true
  }'


EXAMPLE 2: Batch Prediction with Multiple Guides

curl -X POST "http://localhost:8000/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "guides": [
      {"sequence": "ACGTACGTACGTACGTACGTACG"},
      {"sequence": "TGCATGCATGCATGCATGCATGCA"},
      {"sequence": "AAAATTTTCCCCGGGGACGTACGT"},
      {"sequence": "GGGGCCCCAAAAGGGGCCCCAAAA"},
      {"sequence": "TTTTAAAATTTTCCCCGGGGAAAA"}
    ],
    "return_all": false
  }'


EXAMPLE 3: Python Client

import requests
import json

API_URL = "http://localhost:8000"

# Single prediction
response = requests.post(f"{API_URL}/predict", json={
    "sequence": "ACGTACGTACGTACGTACGTACG",
    "cas_type": "cas9",
    "include_uncertainty": True
})

if response.status_code == 200:
    prediction = response.json()
    print(f"Efficiency: {prediction['efficiency_score']:.3f}")
    print(f"Design Score: {prediction['design_score']:.3f}")
    print(f"Safety Tier: {prediction['safety_tier']}")
    if prediction['efficiency_lower']:
        print(f"Efficiency CI: [{prediction['efficiency_lower']:.3f}, {prediction['efficiency_upper']:.3f}]")
else:
    print(f"Error: {response.status_code}")
    print(response.json())


EXAMPLE 4: Batch Prediction with Analysis

import requests
import pandas as pd

guides = ["ACGTACGTACGTACGTACGTACG", "TGCATGCATGCATGCATGCATGCA"]

response = requests.post(f"{API_URL}/predict/batch", json={
    "guides": [{"sequence": seq} for seq in guides],
    "return_all": True
})

if response.status_code == 200:
    results = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "sequence": p["guide_sequence"],
            "efficiency": p["efficiency_score"],
            "off_target_risk": p["off_target_risk"],
            "design_score": p["design_score"],
            "safety_tier": p["safety_tier"]
        }
        for p in results["predictions"]
    ])
    
    # Sort by design score
    df = df.sort_values("design_score", ascending=False)
    print(df)
    
    print(f"\\nProcessing time: {results['processing_time_sec']:.3f}s")
    print(f"High quality guides: {results['num_high_quality']}/{results['num_guides']}")


EXAMPLE 5: Conformal Prediction Intervals

import requests
import numpy as np
import matplotlib.pyplot as plt

response = requests.post(f"{API_URL}/predict", json={
    "sequence": "ACGTACGTACGTACGTACGTACG",
    "include_uncertainty": True
})

pred = response.json()

# Extract interval
lower = pred["efficiency_lower"]
upper = pred["efficiency_upper"]
point = pred["efficiency_score"]

# Plot
fig, ax = plt.subplots()
ax.plot([lower, upper], [0, 0], 'b-', linewidth=2, label='90% Confidence Interval')
ax.plot(point, 0, 'ro', markersize=10, label='Point Prediction')
ax.set_xlim([0, 1])
ax.set_ylim([-0.5, 0.5])
ax.legend()
ax.set_title(f'Efficiency Prediction: {point:.3f} [{lower:.3f}, {upper:.3f}]')
plt.show()


EXAMPLE 6: JavaScript/Fetch Client

async function predictGuide(sequence) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            sequence: sequence,
            cas_type: 'cas9',
            include_uncertainty: true
        })
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
}

// Usage
const result = await predictGuide('ACGTACGTACGTACGTACGTACG');
console.log(`Design Score: ${result.design_score.toFixed(3)}`);
"""

# ==============================================================================
# 4. ERROR HANDLING
# ==============================================================================

"""
STANDARD ERROR RESPONSE FORMAT

{
    "error": "Sequence must contain only ACGT or N",
    "error_type": "validation_error",
    "status_code": 400,
    "details": null
}

COMMON ERROR CODES

400 Bad Request
    - Invalid sequence (non-ACGT characters)
    - Invalid parameter values
    - Missing required fields

401 Unauthorized
    - API key invalid (if authentication enabled)

403 Forbidden
    - Request rate limit exceeded

500 Internal Server Error
    - Model inference error
    - Unexpected server error

503 Service Unavailable
    - Model not loaded
    - GPU out of memory
    - Server startup still in progress

HANDLING ERRORS IN CODE

Python:

try:
    response = requests.post(f"{API_URL}/predict", json=request_data)
    response.raise_for_status()
    results = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        error = e.response.json()
        print(f"Validation error: {error['error']}")
    elif e.response.status_code == 503:
        print("Model not loaded. Try /health endpoint")
    else:
        print(f"HTTP {e.response.status_code}: {error['error']}")
except requests.exceptions.ConnectionError:
    print("Cannot connect to API. Is server running?")

JavaScript:

const handleApiError = (error) => {
    if (error instanceof TypeError) {
        console.error('Network error - API server not running?');
    } else if (error.status === 400) {
        console.error('Invalid input:', error.error);
    } else if (error.status === 503) {
        console.error('Model not loaded');
    } else {
        console.error(`API error ${error.status}:`, error.error);
    }
};
"""

# ==============================================================================
# 5. DEPLOYMENT
# ==============================================================================

"""
LOCAL DEVELOPMENT

1. Install dependencies:
    pip install -r requirements.txt

2. Start server:
    uvicorn src.api.main:app --reload

3. Test endpoint:
    curl http://localhost:8000/health


PRODUCTION DEPLOYMENT - DOCKER

Dockerfile:

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    cuda-toolkit-11-8 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download model checkpoint
RUN python -c "from src.model import create_model; create_model('chromaguide')"

# Expose port
EXPOSE 8000

# Run server
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.api.main:app", "--bind", "0.0.0.0:8000"]


PRODUCTION DEPLOYMENT - KUBERNETES

apiVersion: apps/v1
kind: Deployment
metadata:
  name: chromaguide-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chromaguide-api
  template:
    metadata:
      labels:
        app: chromaguide-api
    spec:
      containers:
      - name: chromaguide-api
        image: chromaguide-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHROMAGUIDE_MODEL
          value: "chromaguide"
        - name: CHROMAGUIDE_CHECKPOINT
          value: "/models/best.pt"
        volumeMounts:
        - name: models
          mountPath: /models
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: chromaguide-models


GUNICORN CONFIGURATION

gunicorn.conf.py:

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count()
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/chromaguide-access.log"
errorlog = "/var/log/chromaguide-error.log"
loglevel = "info"

# Timeouts
timeout = 120
graceful_timeout = 30

# Server mechanics
daemon = False
pidfile = "/var/run/chromaguide.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "/path/to/keyfile.pem"
# certfile = "/path/to/certfile.pem"
"""

# ==============================================================================
# 6. PERFORMANCE OPTIMIZATION
# ==============================================================================

"""
BATCH PROCESSING FOR THROUGHPUT

Instead of:
    for seq in sequences:
        response = requests.post("/predict", json={"sequence": seq})

Do this:
    response = requests.post("/predict/batch", json={
        "guides": [{"sequence": seq} for seq in sequences]
    })

Batch processing is:
- 10-50x faster for large numbers of guides
- Better GPU utilization
- Reduced network overhead


GPU SUPPORT

1. Install CUDA toolkit
2. Install PyTorch with CUDA support:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. Start API (GPU will be auto-detected):
    uvicorn src.api.main:app

4. Check health endpoint:
    curl http://localhost:8000/health
    # Should show "gpu_available": true


CACHING

The API implements:
- Single prediction caching (identical sequences return cached results)
- Model weights are reused across requests
- Warm-up first request (model compilation/loading)

For fastest performance, do a health check first:
    curl http://localhost:8000/health


MONITORING PERFORMANCE

Example monitoring script:

import requests
import time
import statistics

API_URL = "http://localhost:8000"

# Benchmark single prediction
times = []
for _ in range(100):
    start = time.time()
    requests.post(f"{API_URL}/predict", json={"sequence": "ACGTACGTACGTACGTACGTACG"})
    times.append(time.time() - start)

print(f"Single prediction latency:")
print(f"  Mean: {statistics.mean(times)*1000:.2f}ms")
print(f"  Median: {statistics.median(times)*1000:.2f}ms")
print(f"  Std: {statistics.stdev(times)*1000:.2f}ms")

# Benchmark batch prediction
batch_sizes = [10, 50, 100, 500]
for batch_size in batch_sizes:
    guides = [{"sequence": "ACGTACGTACGTACGTACGTACG"} for _ in range(batch_size)]
    start = time.time()
    response = requests.post(f"{API_URL}/predict/batch", json={"guides": guides})
    elapsed = time.time() - start
    throughput = batch_size / elapsed
    print(f"Batch {batch_size}: {throughput:.0f} guides/sec")
"""

# ==============================================================================
# 7. TROUBLESHOOTING
# ==============================================================================

"""
ISSUE: "Model not loaded" or 503 Service Unavailable

Solution 1: Check health endpoint
    curl http://localhost:8000/health
    
    If model_loaded is false, try:
    curl -X POST "http://localhost:8000/models/load?model_name=chromaguide"

Solution 2: Check server logs for errors
    # Check if server is running
    curl http://localhost:8000/
    
Solution 3: Load model with checkpoint
    export CHROMAGUIDE_CHECKPOINT=/path/to/best.pt
    uvicorn src.api.main:app --reload


ISSUE: Out of Memory (OOM) Error

Solutions:
    - Reduce batch size for batch predictions
    - Use CPU instead of GPU:
        export CUDA_VISIBLE_DEVICES=""
    - Use a smaller model variant
    - Upgrade GPU memory


ISSUE: Slow Response Times

Causes and solutions:
    - First request is slow (model loading): This is normal
    - GPU not being used: Check health endpoint, ensure CUDA is available
    - Large batch size: Reduce batch_size parameter
    - Network latency: Use batch predictions instead of many single requests


ISSUE: Invalid sequence error for valid input

Causes:
    - Non-uppercase letters: Sequence is automatically converted to uppercase
    - Non-standard nucleotides: Only ACGT and N are supported
    - Whitespace in sequence: Strip whitespace before sending

Debug:
    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: application/json" \\
      -d '{"sequence": "ACGTACGTACGTACGTACGTACG"}'


ISSUE: API returns different predictions for same sequence

This shouldn't happen. All predictions are deterministic. If it occurs:
    - Check that model is fully loaded (health endpoint)
    - Ensure no other processes are interfering
    - Restart API server


LOGGING & DEBUGGING

Enable debug logging:

import logging
logging.basicConfig(level=logging.DEBUG)

Then check logs for detailed error information.

For server-side debugging, check startup output:

    uvicorn src.api.main:app --log-level debug
"""

# ==============================================================================
# API SUMMARY
# ==============================================================================

"""
QUICK REFERENCE

Health Check:
    GET /health

Single Prediction:
    POST /predict
    {"sequence": "ACGTACGTACGTACGTACGTACG", ...}

Batch Prediction:
    POST /predict/batch
    {"guides": [{"sequence": "..."}, ...]}

Model Info:
    POST /models/info

Load Model:
    POST /models/load?model_name=crispro

Design Score:
    POST /design-score
    {"efficiency_score": 0.8, "off_target_risk": 0.1, ...}


PREDICTIONS INCLUDE:
    - efficiency_score: On-target efficiency (0-1)
    - off_target_risk: Off-target risk (0-1)
    - design_score: Integrated score (0-1)
    - safety_tier: Risk classification
    - activity_probability: Probability of activity
    - specificity_score: Sequence specificity

With uncertainty=true:
    - efficiency_lower/upper: Confidence intervals
    - off_target_lower/upper: Risk confidence intervals
"""

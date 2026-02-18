"""ChromaGuide REST API - FastAPI Application

Endpoints for sgRNA design prediction with on-target efficiency,
off-target risk, and integrated design scoring.

Usage:
    uvicorn src.api.main:app --reload --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import logging
import time
import psutil
import os
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    GuideRNAInput, BatchGuideRNAInput,
    PredictionOutput, BatchPredictionOutput,
    DesignScoreRequest, DesignScoreResponse,
    ModelInfoRequest, ModelsInfoResponse, ModelInfo,
    HealthCheckResponse, ErrorResponse,
)
from .inference import (
    get_inference_engine, reinitialize_engine,
    ChromaGuideInferenceEngine
)
from .middleware import (
    validate_sgrna_sequence, validate_sgrna_batch,
    InvalidSgrnaError,
    RateLimiter, RateLimitExceededError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API versioning
API_VERSION = "1.0.0"
API_TITLE = "ChromaGuide CRISPR sgRNA Design API"

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description="State-of-the-art CRISPR sgRNA design with on-target efficiency, "
                "off-target risk prediction, and conformal uncertainty quantification",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
# Configuration: 100 requests per 60 seconds per IP
rate_limiter = RateLimiter(
    max_requests=100,
    window_seconds=60,
    cleanup_interval=100,
)

# Track startup time
startup_time = None
inference_engine: Optional[ChromaGuideInferenceEngine] = None


# ==============================================================================
# STARTUP & SHUTDOWN EVENTS
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global startup_time, inference_engine
    startup_time = datetime.now()
    
    logger.info(f"{'='*70}")
    logger.info(f"Starting {API_TITLE} (v{API_VERSION})")
    logger.info(f"{'='*70}")
    
    try:
        # Try to load from checkpoint if available
        checkpoint = os.environ.get('CHROMAGUIDE_CHECKPOINT')
        model_name = os.environ.get('CHROMAGUIDE_MODEL', 'chromaguide')
        
        inference_engine = get_inference_engine(
            model_name=model_name,
            checkpoint_path=checkpoint,
        )
        
        logger.info(f"✓ Inference engine initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Checkpoint: {checkpoint or 'using default initialization'}")
        logger.info(f"✓ API ready for predictions")
    
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        logger.warning("API will attempt inference, but model may not be loaded correctly")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ChromaGuide API")


# ==============================================================================
# HEALTH CHECK ENDPOINTS
# ==============================================================================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health & Status"],
    summary="Health check endpoint",
)
async def health_check(req: Request) -> HealthCheckResponse:
    """Check API and model health status.
    
    Returns comprehensive status including:
    - Model loading status
    - GPU availability
    - Memory usage
    - Uptime
    
    Args:
        req: HTTP request (for rate limiting)
    """
    global inference_engine, startup_time
    
    # Get client IP for rate limiting
    client_ip = req.client.host if req.client else "unknown"
    
    # Check rate limit (but let health check through if limit close)
    try:
        rate_limiter.check_rate_limit(client_ip)
    except RateLimitExceededError:
        # Health check is less critical, but still enforce limits
        pass
    
    uptime = (datetime.now() - startup_time).total_seconds() if startup_time else 0
    
    model_loaded = False
    num_predictions = 0
    if inference_engine:
        status = inference_engine.get_status()
        model_loaded = status.get('model_loaded', False)
        num_predictions = status.get('num_predictions', 0)
    
    # Memory usage
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        memory_mb = 0.0
    
    # GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        version=API_VERSION,
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        memory_usage_mb=memory_mb,
        uptime_seconds=uptime,
        details={
            'num_predictions': num_predictions,
            'timestamp': datetime.now().isoformat(),
        }
    )


@app.get(
    "/",
    tags=["Root"],
    summary="API information",
)
async def root():
    """Root endpoint with API information."""
    return {
        "api": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# ==============================================================================
# MODEL INFORMATION ENDPOINTS
# ==============================================================================

@app.post(
    "/models/info",
    response_model=ModelsInfoResponse,
    tags=["Model Management"],
    summary="Get available models",
)
async def get_models_info(request: Optional[ModelInfoRequest] = None) -> ModelsInfoResponse:
    """Get information about available models.
    
    Returns list of models with descriptions and configurations.
    """
    try:
        from src.model import list_available_models, model_info
        
        available = list_available_models()
        model_names = list(available.keys())
        
        # Get detailed info if requested
        models_info = None
        if request and request.model_name:
            try:
                info = model_info(request.model_name)
                models_info = [
                    ModelInfo(
                        name=request.model_name,
                        description=available.get(request.model_name, ""),
                        Parameters=info.get('approx_params'),
                    )
                ]
            except Exception as e:
                logger.warning(f"Could not get info for model {request.model_name}: {e}")
        
        return ModelsInfoResponse(
            available_models=model_names,
            models_info=models_info,
        )
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/models/load",
    tags=["Model Management"],
    summary="Load a specific model",
)
async def load_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Load a specific model by name.
    
    Args:
        model_name: Model identifier
        checkpoint_path: Optional checkpoint path
        
    Returns:
        Status message with model info
    """
    global inference_engine
    
    try:
        inference_engine = reinitialize_engine(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )
        
        status = inference_engine.get_status()
        return {
            "status": "success",
            "message": f"Loaded model '{model_name}'",
            "model_info": status,
        }
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# PREDICTION ENDPOINTS
# ==============================================================================

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Predictions"],
    summary="Predict for single sgRNA",
)
async def predict_single(request: GuideRNAInput, req: Request) -> PredictionOutput:
    """Predict on-target efficiency and off-target risk for a single sgRNA.
    
    Args:
        request: GuideRNA prediction request
        req: HTTP request (for rate limiting)
        
    Returns:
        Prediction with efficiency, off-target risk, design score, and safety tier
    """
    global inference_engine
    
    # Get client IP for rate limiting
    client_ip = req.client.host if req.client else "unknown"
    
    # Check rate limit
    try:
        rate_limiter.check_rate_limit(client_ip)
    except RateLimitExceededError as e:
        remaining_time = int(rate_limiter.get_window_reset_time(client_ip) - time.time())
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {str(e)}. Retry after {remaining_time}s",
        )
    
    # Validate sgRNA sequence
    try:
        validate_sgrna_sequence(request.sequence)
    except InvalidSgrnaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not inference_engine or not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /health or /models/load first.")
    
    try:
        # Run prediction
        prediction = inference_engine.predict_single(
            sequence=request.sequence,
            include_uncertainty=request.include_uncertainty,
        )
        
        # Compute design score
        design_result = inference_engine.compute_design_score(
            efficiency=prediction['efficiency'],
            off_target_risk=prediction['off_target'],
        )
        
        # Determine safety tier
        safety_score = 1 - prediction['off_target']
        if safety_score >= 0.9:
            safety_tier = "very_low_risk"
        elif safety_score >= 0.7:
            safety_tier = "low_risk"
        elif safety_score >= 0.5:
            safety_tier = "medium_risk"
        elif safety_score >= 0.3:
            safety_tier = "high_risk"
        else:
            safety_tier = "very_high_risk"
        
        # Activity probability (simple heuristic)
        activity_prob = prediction['efficiency']
        
        # Specificity (proxy from biophysics if available)
        specificity = prediction.get('delta_g', 0.5)
        if isinstance(specificity, (int, float)):
            # Normalize delta_g to [0, 1] if available
            specificity = np.clip(specificity / 20, 0, 1)
        else:
            specificity = 0.5
        
        return PredictionOutput(
            guide_sequence=request.sequence,
            efficiency_score=prediction['efficiency'],
            efficiency_lower=prediction.get('efficiency_lower'),
            efficiency_upper=prediction.get('efficiency_upper'),
            off_target_risk=prediction['off_target'],
            off_target_lower=prediction.get('off_target_lower'),
            off_target_upper=prediction.get('off_target_upper'),
            design_score=design_result['design_score'],
            safety_tier=safety_tier,
            activity_probability=activity_prob,
            specificity_score=specificity,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    summary="Predict for multiple sgRNAs",
)
async def predict_batch(request: BatchGuideRNAInput, req: Request) -> BatchPredictionOutput:
    """Predict for multiple sgRNAs in a batch.
    
    Args:
        request: Batch prediction request
        req: HTTP request (for rate limiting)
        
    Returns:
        Batch predictions with optional top-k filtering
    """
    global inference_engine
    
    # Get client IP for rate limiting
    client_ip = req.client.host if req.client else "unknown"
    
    # Check rate limit
    try:
        rate_limiter.check_rate_limit(client_ip)
    except RateLimitExceededError as e:
        remaining_time = int(rate_limiter.get_window_reset_time(client_ip) - time.time())
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {str(e)}. Retry after {remaining_time}s",
        )
    
    # Validate all sgRNA sequences
    sequences = [guide.sequence for guide in request.guides]
    valid_sequences, error_messages = validate_sgrna_batch(sequences)
    
    # Check for validation errors
    errors = [(i, msg) for i, msg in enumerate(error_messages) if msg]
    if errors:
        error_details = "\n".join([f"  Sequence {i}: {msg}" for i, msg in errors])
        raise HTTPException(
            status_code=400,
            detail=f"Validation failed for {len(errors)} sequences:\n{error_details}",
        )
    
    if not inference_engine or not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    start_time = time.time()
    
    try:
        # Extract sequences
        sequences = [guide.sequence for guide in request.guides]
        
        # Run batch prediction
        predictions = inference_engine.predict_batch(
            sequences=sequences,
            include_uncertainty=request.guides[0].include_uncertainty if request.guides else True,
        )
        
        # Build output
        prediction_outputs = []
        for guide, pred in zip(request.guides, predictions):
            # Compute design score
            design_result = inference_engine.compute_design_score(
                efficiency=pred['efficiency'],
                off_target_risk=pred['off_target'],
            )
            
            # Safety tier
            safety_score = 1 - pred['off_target']
            if safety_score >= 0.9:
                safety_tier = "very_low_risk"
            elif safety_score >= 0.7:
                safety_tier = "low_risk"
            elif safety_score >= 0.5:
                safety_tier = "medium_risk"
            elif safety_score >= 0.3:
                safety_tier = "high_risk"
            else:
                safety_tier = "very_high_risk"
            
            specificity = pred.get('delta_g', 0.5)
            if isinstance(specificity, (int, float)):
                specificity = np.clip(specificity / 20, 0, 1)
            else:
                specificity = 0.5
            
            output = PredictionOutput(
                guide_sequence=guide.sequence,
                efficiency_score=pred['efficiency'],
                efficiency_lower=pred.get('efficiency_lower'),
                efficiency_upper=pred.get('efficiency_upper'),
                off_target_risk=pred['off_target'],
                off_target_lower=pred.get('off_target_lower'),
                off_target_upper=pred.get('off_target_upper'),
                design_score=design_result['design_score'],
                safety_tier=safety_tier,
                activity_probability=pred['efficiency'],
                specificity_score=specificity,
            )
            prediction_outputs.append(output)
        
        # Sort by design score
        prediction_outputs.sort(key=lambda x: x.design_score, reverse=True)
        
        # Filter top guides if requested
        top_guides = None
        if not request.return_all:
            top_guides = prediction_outputs[:10]
        
        processing_time = time.time() - start_time
        
        return BatchPredictionOutput(
            predictions=prediction_outputs,
            num_guides=len(request.guides),
            num_high_quality=len([p for p in prediction_outputs if p.design_score > 0.5]),
            top_guides=top_guides,
            processing_time_sec=processing_time,
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# UTILITY ENDPOINTS
# ==============================================================================

@app.post(
    "/design-score",
    response_model=DesignScoreResponse,
    tags=["Utilities"],
    summary="Calculate design score",
)
async def calculate_design_score(request: DesignScoreRequest, req: Request) -> DesignScoreResponse:
    """Calculate integrated design score from components.
    
    Args:
        request: Design score request
        req: HTTP request (for rate limiting)
        
    Returns:
        Design score and component breakdown
    """
    global inference_engine
    
    # Get client IP for rate limiting
    client_ip = req.client.host if req.client else "unknown"
    
    # Check rate limit
    try:
        rate_limiter.check_rate_limit(client_ip)
    except RateLimitExceededError as e:
        remaining_time = int(rate_limiter.get_window_reset_time(client_ip) - time.time())
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {str(e)}. Retry after {remaining_time}s",
        )
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    
    try:
        result = inference_engine.compute_design_score(
            efficiency=request.efficiency_score,
            off_target_risk=request.off_target_risk,
            specificity=request.specificity_score,
            eff_weight=request.efficiency_weight,
            safety_weight=request.safety_weight,
        )
        
        return DesignScoreResponse(
            design_score=result['design_score'],
            components=result['components'],
        )
    
    except Exception as e:
        logger.error(f"Design score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_type": "http_error",
            "status_code": exc.status_code,
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "error_type": "validation_error",
            "status_code": 400,
        }
    )


# ==============================================================================
# IMPORTS FOR NUMPY IN RESPONSES
# ==============================================================================

import numpy as np

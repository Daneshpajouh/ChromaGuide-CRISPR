"""Pydantic models for ChromaGuide API.

Defines all request and response schemas for the REST API.
Includes validation for DNA sequences, guides, and predictions.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import re


class GuideRNAInput(BaseModel):
    """Single sgRNA prediction request."""
    
    sequence: str = Field(
        ...,
        description="Target sequence (DNA, 20-25bp for Cas9, 18-25bp for Cas12a)",
        min_length=18,
        max_length=30,
    )
    cas_type: str = Field(
        default="cas9",
        description="CRISPR-Cas type: 'cas9' or 'cas12a'",
        regex="^(cas9|cas12a)$",
    )
    off_target_threshold: float = Field(
        default=0.1,
        description="Off-target risk threshold (0-1)",
        ge=0.0,
        le=1.0,
    )
    include_uncertainty: bool = Field(
        default=True,
        description="Include conformal prediction intervals",
    )
    include_explanations: bool = Field(
        default=False,
        description="Include feature importance and attention maps",
    )
    genome_version: str = Field(
        default="hg38",
        description="Reference genome version",
    )
    
    @validator('sequence')
    def validate_sequence(cls, v):
        """Validate DNA sequence characters."""
        valid_chars = set('ACGTNacgtn')
        if not all(c in valid_chars for c in v):
            raise ValueError('Sequence must contain only ACGT or N')
        return v.upper()


class BatchGuideRNAInput(BaseModel):
    """Batch sgRNA prediction request."""
    
    guides: List[GuideRNAInput] = Field(
        ...,
        description="List of guide RNA sequences",
        min_items=1,
        max_items=1000,
    )
    return_all: bool = Field(
        default=False,
        description="Return predictions for all guides (vs. top-k filtered)",
    )


class PredictionOutput(BaseModel):
    """Single sgRNA prediction response."""
    
    guide_sequence: str = Field(
        ...,
        description="Input guide sequence",
    )
    efficiency_score: float = Field(
        ...,
        description="Predicted on-target efficiency (0-1)",
        ge=0.0,
        le=1.0,
    )
    efficiency_lower: Optional[float] = Field(
        None,
        description="Lower bound of conformal prediction interval",
    )
    efficiency_upper: Optional[float] = Field(
        None,
        description="Upper bound of conformal prediction interval",
    )
    off_target_risk: float = Field(
        ...,
        description="Predicted off-target risk score (0-1)",
        ge=0.0,
        le=1.0,
    )
    off_target_lower: Optional[float] = Field(
        None,
        description="Lower bound of off-target conformal interval",
    )
    off_target_upper: Optional[float] = Field(
        None,
        description="Upper bound of off-target conformal interval",
    )
    design_score: float = Field(
        ...,
        description="Integrated design score balancing efficiency and safety (0-1)",
        ge=0.0,
        le=1.0,
    )
    safety_tier: str = Field(
        ...,
        description="Safety classification: 'very_high_risk', 'high_risk', 'medium_risk', 'low_risk', 'very_low_risk'",
    )
    activity_probability: float = Field(
        ...,
        description="Probability of guide being active (0-1)",
        ge=0.0,
        le=1.0,
    )
    specificity_score: float = Field(
        ...,
        description="Specificity score based on sequence conservation (0-1)",
        ge=0.0,
        le=1.0,
    )


class BatchPredictionOutput(BaseModel):
    """Batch prediction response."""
    
    predictions: List[PredictionOutput] = Field(
        ...,
        description="List of predictions for each guide",
    )
    num_guides: int = Field(
        ...,
        description="Total number of guides processed",
    )
    num_high_quality: int = Field(
        ...,
        description="Number of guides passing quality filters",
    )
    top_guides: Optional[List[PredictionOutput]] = Field(
        None,
        description="Top-k highest design score guides (if return_all=False)",
    )
    processing_time_sec: float = Field(
        ...,
        description="Total processing time in seconds",
    )


class DesignScoreRequest(BaseModel):
    """Request for custom design score calculation."""
    
    efficiency_score: float = Field(
        ...,
        description="On-target efficiency (0-1)",
        ge=0.0,
        le=1.0,
    )
    off_target_risk: float = Field(
        ...,
        description="Off-target risk (0-1)",
        ge=0.0,
        le=1.0,
    )
    specificity_score: float = Field(
        ...,
        description="Specificity score (0-1)",
        ge=0.0,
        le=1.0,
    )
    efficiency_weight: float = Field(
        default=0.5,
        description="Weight for efficiency in design score",
        ge=0.0,
        le=1.0,
    )
    safety_weight: float = Field(
        default=0.5,
        description="Weight for safety (1-off_target_risk) in design score",
        ge=0.0,
        le=1.0,
    )


class DesignScoreResponse(BaseModel):
    """Response for design score calculation."""
    
    design_score: float = Field(
        ...,
        description="Calculated design score (0-1)",
    )
    components: Dict[str, float] = Field(
        ...,
        description="Score components breakdown",
    )


class ModelInfoRequest(BaseModel):
    """Request for model information."""
    
    model_name: Optional[str] = Field(
        None,
        description="Specific model name, or all if not specified",
    )


class ModelInfo(BaseModel):
    """Information about a model."""
    
    name: str = Field(
        ...,
        description="Model identifier",
    )
    description: str = Field(
        ...,
        description="Model description",
    )
    parameters: Optional[int] = Field(
        None,
        description="Number of trainable parameters",
    )
    input_dimensions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected input dimensions",
    )
    default_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default configuration",
    )


class ModelsInfoResponse(BaseModel):
    """Response with available models information."""
    
    available_models: List[str] = Field(
        ...,
        description="List of available model names",
    )
    models_info: Optional[List[ModelInfo]] = Field(
        None,
        description="Detailed info for each model",
    )


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        default="healthy",
        description="Service status: 'healthy', 'degraded', 'unhealthy'",
    )
    version: str = Field(
        ...,
        description="API version",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether primary model is loaded",
    )
    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available",
    )
    memory_usage_mb: float = Field(
        ...,
        description="Memory usage in MB",
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional status details",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(
        ...,
        description="Error message",
    )
    error_type: str = Field(
        ...,
        description="Error type: 'validation_error', 'model_error', 'server_error', etc.",
    )
    status_code: int = Field(
        ...,
        description="HTTP status code",
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details",
    )


class ExplanationOutput(BaseModel):
    """Model explanation for a prediction."""
    
    guide_sequence: str = Field(
        ...,
        description="Input guide sequence",
    )
    motif_importance: Dict[str, float] = Field(
        ...,
        description="Importance scores for sequence motifs",
    )
    position_importance: List[float] = Field(
        ...,
        description="Per-position importance scores",
    )
    attention_map: Optional[List[List[float]]] = Field(
        None,
        description="Attention weights if available",
    )
    top_features: List[Dict[str, Any]] = Field(
        ...,
        description="Top contributing features",
    )
    predicted_efficiency: float = Field(
        ...,
        description="Predicted efficiency for this guide",
    )


class PredictionWithExplanation(PredictionOutput):
    """Prediction with explanation."""
    
    explanation: Optional[ExplanationOutput] = Field(
        None,
        description="Model explanation (if requested)",
    )


class BatchPredictionWithExplanations(BatchPredictionOutput):
    """Batch predictions with explanations."""
    
    predictions: List[PredictionWithExplanation] = Field(
        ...,
        description="List of predictions with optional explanations",
    )

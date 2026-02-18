"""
ChromaGuide Python Client Library

Easy-to-use client for interacting with ChromaGuide REST API.

Example:
    from chromaguide_client import ChromaGuideClient
    
    client = ChromaGuideClient(api_url="http://localhost:8000")
    
    # Single prediction
    result = client.predict_single("ACGTACGTACGTACGTACGTACG")
    print(f"Design Score: {result.design_score}")
    
    # Batch prediction
    sequences = ["ACGTACGTACGTACGTACGTACG", "TGCATGCATGCATGCATGCATGCA"]
    results = client.predict_batch(sequences)
    for r in results:
        print(f"{r.guide_sequence}: {r.design_score:.3f}")
"""

import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json


class SafetyTier(str, Enum):
    """Safety classification for guides."""
    VERY_LOW_RISK = "very_low_risk"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    VERY_HIGH_RISK = "very_high_risk"


class CasType(str, Enum):
    """CRISPR-Cas variants."""
    CAS9 = "cas9"
    CAS12A = "cas12a"


@dataclass
class PredictionResult:
    """Single guide prediction result."""
    guide_sequence: str
    efficiency_score: float
    off_target_risk: float
    design_score: float
    safety_tier: SafetyTier
    activity_probability: float
    specificity_score: float
    efficiency_lower: Optional[float] = None
    efficiency_upper: Optional[float] = None
    off_target_lower: Optional[float] = None
    off_target_upper: Optional[float] = None
    
    def __str__(self):
        """String representation."""
        return (
            f"Guide: {self.guide_sequence}\n"
            f"  Design Score:     {self.design_score:.3f}\n"
            f"  Efficiency:       {self.efficiency_score:.3f}"
            f" [{self.efficiency_lower:.3f}, {self.efficiency_upper:.3f}]\n"
            f"  Off-target Risk:  {self.off_target_risk:.3f}"
            f" [{self.off_target_lower:.3f}, {self.off_target_upper:.3f}]\n"
            f"  Activity Prob:    {self.activity_probability:.3f}\n"
            f"  Specificity:      {self.specificity_score:.3f}\n"
            f"  Safety Tier:      {self.safety_tier}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sequence': self.guide_sequence,
            'design_score': self.design_score,
            'efficiency': self.efficiency_score,
            'off_target_risk': self.off_target_risk,
            'safety_tier': self.safety_tier,
        }


@dataclass
class BatchPredictionResult:
    """Batch prediction results."""
    predictions: List[PredictionResult]
    num_guides: int
    num_high_quality: int
    processing_time_sec: float
    top_guides: Optional[List[PredictionResult]] = None
    
    def __str__(self):
        """String representation."""
        lines = [
            f"Batch Results ({self.num_guides} guides)",
            f"  High Quality:    {self.num_high_quality}",
            f"  Processing Time: {self.processing_time_sec:.3f}s",
            f"  Throughput:      {self.num_guides/self.processing_time_sec:.0f} guides/sec",
        ]
        
        if self.top_guides:
            lines.append(f"\n  Top Guides (by design score):")
            for i, pred in enumerate(self.top_guides[:5], 1):
                lines.append(
                    f"    {i}. {pred.guide_sequence}: "
                    f"design={pred.design_score:.3f}, "
                    f"safety={pred.safety_tier}"
                )
        
        return "\n".join(lines)


class ChromaGuideClient:
    """Client for ChromaGuide REST API."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """Initialize client.
        
        Args:
            api_url: Base URL for API
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = requests.Session()
        self._session.verify = verify_ssl
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without leading /)
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response
            
        Raises:
            requests.RequestException: Network errors
            ValueError: API errors
        """
        url = f"{self.api_url}/{endpoint}"
        
        try:
            response = self._session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to API at {self.api_url}: {e}")
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"API request timed out after {self.timeout}s: {e}")
        except requests.exceptions.HTTPError as e:
            error_data = response.json() if response.text else {}
            raise ValueError(
                f"API error {response.status_code}: "
                f"{error_data.get('error', response.reason)}"
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status.
        
        Returns:
            Health status dict
        """
        return self._request("GET", "health")
    
    def predict_single(
        self,
        sequence: str,
        cas_type: CasType = CasType.CAS9,
        include_uncertainty: bool = True,
        off_target_threshold: float = 0.1,
    ) -> PredictionResult:
        """Predict for single guide RNA.
        
        Args:
            sequence: DNA sequence
            cas_type: Cas variant
            include_uncertainty: Include confidence intervals
            off_target_threshold: Off-target risk threshold
            
        Returns:
            PredictionResult
        """
        response = self._request(
            "POST",
            "predict",
            json={
                "sequence": sequence,
                "cas_type": cas_type.value,
                "include_uncertainty": include_uncertainty,
                "off_target_threshold": off_target_threshold,
            }
        )
        
        return PredictionResult(
            guide_sequence=response['guide_sequence'],
            efficiency_score=response['efficiency_score'],
            off_target_risk=response['off_target_risk'],
            design_score=response['design_score'],
            safety_tier=SafetyTier(response['safety_tier']),
            activity_probability=response['activity_probability'],
            specificity_score=response['specificity_score'],
            efficiency_lower=response.get('efficiency_lower'),
            efficiency_upper=response.get('efficiency_upper'),
            off_target_lower=response.get('off_target_lower'),
            off_target_upper=response.get('off_target_upper'),
        )
    
    def predict_batch(
        self,
        sequences: List[str],
        cas_type: CasType = CasType.CAS9,
        include_uncertainty: bool = True,
        return_all: bool = False,
    ) -> BatchPredictionResult:
        """Predict for multiple guides.
        
        Args:
            sequences: List of DNA sequences
            cas_type: Cas variant
            include_uncertainty: Include confidence intervals
            return_all: Return all predictions or just top-k
            
        Returns:
            BatchPredictionResult
        """
        response = self._request(
            "POST",
            "predict/batch",
            json={
                "guides": [
                    {
                        "sequence": seq,
                        "cas_type": cas_type.value,
                        "include_uncertainty": include_uncertainty,
                    }
                    for seq in sequences
                ],
                "return_all": return_all,
            }
        )
        
        predictions = [
            PredictionResult(
                guide_sequence=p['guide_sequence'],
                efficiency_score=p['efficiency_score'],
                off_target_risk=p['off_target_risk'],
                design_score=p['design_score'],
                safety_tier=SafetyTier(p['safety_tier']),
                activity_probability=p['activity_probability'],
                specificity_score=p['specificity_score'],
                efficiency_lower=p.get('efficiency_lower'),
                efficiency_upper=p.get('efficiency_upper'),
                off_target_lower=p.get('off_target_lower'),
                off_target_upper=p.get('off_target_upper'),
            )
            for p in response['predictions']
        ]
        
        top_guides = None
        if response.get('top_guides'):
            top_guides = [
                PredictionResult(
                    guide_sequence=p['guide_sequence'],
                    efficiency_score=p['efficiency_score'],
                    off_target_risk=p['off_target_risk'],
                    design_score=p['design_score'],
                    safety_tier=SafetyTier(p['safety_tier']),
                    activity_probability=p['activity_probability'],
                    specificity_score=p['specificity_score'],
                )
                for p in response['top_guides']
            ]
        
        return BatchPredictionResult(
            predictions=predictions,
            num_guides=response['num_guides'],
            num_high_quality=response['num_high_quality'],
            processing_time_sec=response['processing_time_sec'],
            top_guides=top_guides,
        )
    
    def calculate_design_score(
        self,
        efficiency: float,
        off_target_risk: float,
        specificity: float = 0.5,
        efficiency_weight: float = 0.5,
        safety_weight: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate design score.
        
        Args:
            efficiency: On-target efficiency (0-1)
            off_target_risk: Off-target risk (0-1)
            specificity: Specificity score (0-1)
            efficiency_weight: Weight for efficiency
            safety_weight: Weight for safety
            
        Returns:
            Dict with design_score and components
        """
        return self._request(
            "POST",
            "design-score",
            json={
                "efficiency_score": efficiency,
                "off_target_risk": off_target_risk,
                "specificity_score": specificity,
                "efficiency_weight": efficiency_weight,
                "safety_weight": safety_weight,
            }
        )
    
    def list_models(self) -> List[str]:
        """List available models.
        
        Returns:
            List of model names
        """
        response = self._request("POST", "models/info", json={})
        return response['available_models']
    
    def load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a specific model.
        
        Args:
            model_name: Model identifier
            checkpoint_path: Optional checkpoint path
            
        Returns:
            Status dict
        """
        params = {"model_name": model_name}
        if checkpoint_path:
            params["checkpoint_path"] = checkpoint_path
        
        return self._request("POST", "models/load", params=params)
    
    def close(self):
        """Close client session."""
        self._session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


# Convenience functions

def predict_single(
    sequence: str,
    api_url: str = "http://localhost:8000",
    **kwargs
) -> PredictionResult:
    """Quick prediction without creating client.
    
    Args:
        sequence: DNA sequence
        api_url: API base URL
        **kwargs: Additional arguments for predict_single()
        
    Returns:
        PredictionResult
    """
    client = ChromaGuideClient(api_url)
    try:
        return client.predict_single(sequence, **kwargs)
    finally:
        client.close()


def predict_batch(
    sequences: List[str],
    api_url: str = "http://localhost:8000",
    **kwargs
) -> BatchPredictionResult:
    """Quick batch prediction without creating client.
    
    Args:
        sequences: List of DNA sequences
        api_url: API base URL
        **kwargs: Additional arguments for predict_batch()
        
    Returns:
        BatchPredictionResult
    """
    client = ChromaGuideClient(api_url)
    try:
        return client.predict_batch(sequences, **kwargs)
    finally:
        client.close()


if __name__ == '__main__':
    # Example usage
    print("ChromaGuide Python Client Example\n")
    
    # Single prediction
    print("Single Prediction:")
    print("-" * 50)
    try:
        result = predict_single("ACGTACGTACGTACGTACGTACG")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Batch prediction
    print("\n\nBatch Prediction:")
    print("-" * 50)
    try:
        sequences = [
            "ACGTACGTACGTACGTACGTACG",
            "TGCATGCATGCATGCATGCATGCA",
            "AAAATTTTCCCCGGGGACGTACGT",
        ]
        results = predict_batch(sequences)
        print(results)
    except Exception as e:
        print(f"Error: {e}")

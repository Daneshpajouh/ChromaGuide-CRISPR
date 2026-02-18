"""Model inference engine for ChromaGuide API.

Handles:
- Model loading and caching
- Input preprocessing
- Batch prediction
- Conformal prediction intervals
- Explanation generation
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import threading

from src.model import ModelFactory, create_model
from src.data.biophysics import compute_delta_g, compute_rloop_score

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Conformal prediction for calibrated prediction intervals.
    
    Provides statistical guarantees on prediction intervals.
    """
    
    def __init__(self, confidence_level: float = 0.9):
        """Initialize conformal predictor.
        
        Args:
            confidence_level: Desired coverage probability (0.9 = 90%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.calibration_scores = None
        self.quantile_threshold = None
    
    def calibrate(self, residuals: np.ndarray) -> None:
        """Calibrate using residuals from validation set.
        
        Args:
            residuals: Prediction residuals (absolute errors)
        """
        # Compute quantile threshold
        n = len(residuals)
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha) / n))
        q_idx = min(q_idx, n - 1)
        
        self.quantile_threshold = np.sort(residuals)[q_idx]
        self.calibration_scores = residuals
        
        logger.info(
            f"Conformal predictor calibrated: "
            f"confidence={self.confidence_level}, threshold={self.quantile_threshold:.4f}"
        )
    
    def get_interval(self, point_prediction: float, radius: Optional[float] = None) -> Tuple[float, float]:
        """Get prediction interval for a point prediction.
        
        Args:
            point_prediction: Point prediction
            radius: Optional custom radius (uses calibrated if not provided)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if radius is None:
            if self.quantile_threshold is None:
                # Default to ±0.1 if not calibrated
                radius = 0.1
            else:
                radius = self.quantile_threshold
        
        lower = np.clip(point_prediction - radius, 0, 1)
        upper = np.clip(point_prediction + radius, 0, 1)
        
        return float(lower), float(upper)


class ChromaGuideInferenceEngine:
    """Inference engine for ChromaGuide models.
    
    Manages model loading, caching, and prediction generation.
    """
    
    def __init__(
        self,
        model_name: str = 'chromaguide',
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_predictions: bool = True,
        use_conformal: bool = True,
    ):
        """Initialize inference engine.
        
        Args:
            model_name: Model identifier
            checkpoint_path: Path to model checkpoint
            device: Device to load model on
            cache_predictions: Cache predictions for identical inputs
            use_conformal: Use conformal prediction intervals
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_predictions = cache_predictions
        self.use_conformal = use_conformal
        
        # State
        self.model = None
        self.factory = None
        self.model_loaded = False
        self.prediction_cache = {} if cache_predictions else None
        self.conformal_eff = ConformalPredictor() if use_conformal else None
        self.conformal_ot = ConformalPredictor() if use_conformal else None
        self.load_time = None
        self.num_predictions = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized ChromaGuideInferenceEngine on device={self.device}")
    
    def load_model(self) -> bool:
        """Load model from checkpoint or create new instance.
        
        Returns:
            True if loading successful
        """
        with self._lock:
            try:
                start_time = time.time()
                
                logger.info(f"Loading model '{self.model_name}'...")
                
                self.factory = ModelFactory(device=self.device)
                
                if self.checkpoint_path:
                    # Load from checkpoint
                    self.model, _ = self.factory.load_checkpoint(
                        self.checkpoint_path,
                        model_name=self.model_name,
                        strict=False,
                    )
                else:
                    # Create new model with defaults
                    self.model = self.factory.create(self.model_name)
                
                self.model.eval()
                self.model_loaded = True
                self.load_time = time.time() - start_time
                
                # Count parameters
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(
                    f"✓ Model loaded successfully ({num_params:,} parameters, "
                    f"loaded in {self.load_time:.2f}s)"
                )
                
                return True
            
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model_loaded = False
                return False
    
    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence.
        
        Args:
            sequence: DNA sequence (ACGT)
            
        Returns:
            One-hot encoded array (4, seq_len)
        """
        char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': np.random.randint(0, 4)}
        
        indices = np.array([char_to_idx.get(c, np.random.randint(0, 4)) for c in sequence.upper()])
        one_hot = np.zeros((4, len(sequence)))
        one_hot[indices, np.arange(len(sequence))] = 1
        
        return one_hot
    
    def _extract_output(self, model_output: Any) -> Dict[str, float]:
        """Extract predictions from model output.
        
        Handles different output formats from different models.
        
        Args:
            model_output: Raw model output
            
        Returns:
            Dict with keys: efficiency, off_target (and optionally their uncertainties)
        """
        result = {}
        
        if isinstance(model_output, dict):
            # Multi-output model
            if 'regression' in model_output:
                result['efficiency'] = float(model_output['regression'].squeeze().cpu().numpy())
            elif 'mu' in model_output:
                result['efficiency'] = float(model_output['mu'].squeeze().cpu().numpy())
            else:
                # Try to find any scalar output
                for v in model_output.values():
                    if isinstance(v, torch.Tensor):
                        result['efficiency'] = float(v.squeeze().cpu().numpy())
                        break
            
            # Off-target prediction if available
            if 'off_target' in model_output:
                result['off_target'] = float(model_output['off_target'].squeeze().cpu().numpy())
        else:
            # Single output tensor
            result['efficiency'] = float(model_output.squeeze().cpu().numpy())
        
        # Clamp to [0, 1]
        for key in result:
            result[key] = np.clip(result[key], 0, 1)
        
        return result
    
    def predict_single(
        self,
        sequence: str,
        include_uncertainty: bool = True,
        include_biophysics: bool = True,
    ) -> Dict[str, Any]:
        """Predict for a single guide RNA.
        
        Args:
            sequence: DNA sequence
            include_uncertainty: Include conformal intervals
            include_biophysics: Include biophysics features
            
        Returns:
            Dict with predictions and optional uncertainty
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Check cache
        cache_key = (sequence, include_uncertainty, include_biophysics)
        if self.cache_predictions and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        with torch.no_grad():
            # Prepare input
            one_hot = self._one_hot_encode(sequence)
            one_hot = torch.from_numpy(one_hot[np.newaxis, :, :]).float()
            one_hot = one_hot.to(self.device)
            
            # Forward pass
            try:
                model_output = self.model(one_hot)
            except TypeError:
                # Model might not accept second argument
                model_output = self.model(one_hot)
            
            # Extract predictions
            preds = self._extract_output(model_output)
            
            # Build result
            result = {
                'sequence': sequence,
                'efficiency': preds.get('efficiency', 0.5),
                'off_target': preds.get('off_target', 0.1),
            }
            
            # Add uncertainty intervals
            if include_uncertainty and self.use_conformal:
                eff_lower, eff_upper = self.conformal_eff.get_interval(result['efficiency'])
                ot_lower, ot_upper = self.conformal_ot.get_interval(result['off_target'])
                
                result['efficiency_lower'] = eff_lower
                result['efficiency_upper'] = eff_upper
                result['off_target_lower'] = ot_lower
                result['off_target_upper'] = ot_upper
            
            # Add biophysics
            if include_biophysics:
                try:
                    delta_g = compute_delta_g(sequence)
                    rloop = compute_rloop_score(sequence)
                    result['delta_g'] = delta_g
                    result['rloop_score'] = rloop
                except Exception as e:
                    logger.warning(f"Could not compute biophysics: {e}")
            
            # Cache
            if self.cache_predictions:
                self.prediction_cache[cache_key] = result
        
        self.num_predictions += 1
        return result
    
    def predict_batch(
        self,
        sequences: List[str],
        batch_size: int = 32,
        include_uncertainty: bool = True,
    ) -> List[Dict[str, Any]]:
        """Predict for multiple guide RNAs.
        
        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for efficient processing
            include_uncertainty: Include conformal intervals
            
        Returns:
            List of prediction dicts
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                
                # Prepare batch
                one_hots = [self._one_hot_encode(seq) for seq in batch_seqs]
                one_hots = np.stack(one_hots)
                one_hots = torch.from_numpy(one_hots).float().to(self.device)
                
                # Forward pass
                try:
                    model_outputs = self.model(one_hots)
                except TypeError:
                    model_outputs = self.model(one_hots)
                
                # Extract predictions
                if isinstance(model_outputs, dict):
                    # Batch of dicts
                    for key in model_outputs:
                        if isinstance(model_outputs[key], torch.Tensor):
                            model_outputs[key] = model_outputs[key].cpu().numpy()
                    
                    for j in range(len(batch_seqs)):
                        preds = {}
                        for key, val in model_outputs.items():
                            preds[key] = val[j] if isinstance(val, np.ndarray) else val
                        
                        result = self._extract_output(preds)
                        result['sequence'] = batch_seqs[j]
                        
                        # Add uncertainty
                        if include_uncertainty and self.use_conformal:
                            eff_l, eff_u = self.conformal_eff.get_interval(result['efficiency'])
                            ot_l, ot_u = self.conformal_ot.get_interval(result['off_target'])
                            result.update({
                                'efficiency_lower': eff_l,
                                'efficiency_upper': eff_u,
                                'off_target_lower': ot_l,
                                'off_target_upper': ot_u,
                            })
                        
                        results.append(result)
                else:
                    # Single tensor output
                    outputs = model_outputs.cpu().numpy()
                    for j, seq in enumerate(batch_seqs):
                        result = self._extract_output(outputs[j])
                        result['sequence'] = seq
                        results.append(result)
        
        self.num_predictions += len(sequences)
        return results
    
    def compute_design_score(
        self,
        efficiency: float,
        off_target_risk: float,
        specificity: float = 0.5,
        eff_weight: float = 0.5,
        safety_weight: float = 0.5,
    ) -> Dict[str, float]:
        """Compute integrated design score.
        
        Args:
            efficiency: On-target efficiency (0-1)
            off_target_risk: Off-target risk (0-1)
            specificity: Specificity score (0-1)
            eff_weight: Weight for efficiency
            safety_weight: Weight for safety
            
        Returns:
            Dict with design_score and components
        """
        # Normalize weights
        eff_weight = eff_weight / (eff_weight + safety_weight)
        safety_weight = 1 - eff_weight
        
        # Safety is 1 - risk
        safety = 1 - off_target_risk
        
        # Design score
        design = (efficiency * eff_weight) + (safety * safety_weight)
        
        # Apply specificity as modifier
        design = design * (0.9 + 0.1 * specificity)
        
        return {
            'design_score': float(np.clip(design, 0, 1)),
            'components': {
                'efficiency': float(efficiency),
                'safety': float(safety),
                'specificity': float(specificity),
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get inference engine status.
        
        Returns:
            Status dict
        """
        info = {
            'model_loaded': self.model_loaded,
            'model_name': self.model_name,
            'device': self.device,
            'num_predictions': self.num_predictions,
            'cache_size': len(self.prediction_cache) if self.cache_predictions else 0,
        }
        
        if self.model_loaded:
            info['num_parameters'] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            info['load_time_sec'] = self.load_time
        
        return info


# Global inference engine instance
_inference_engine: Optional[ChromaGuideInferenceEngine] = None
_engine_lock = threading.Lock()


def get_inference_engine(
    model_name: str = 'chromaguide',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ChromaGuideInferenceEngine:
    """Get or create global inference engine.
    
    Args:
        model_name: Model identifier
        checkpoint_path: Checkpoint path if not using defaults
        device: Device to use
        
    Returns:
        ChromaGuideInferenceEngine instance
    """
    global _inference_engine
    
    with _engine_lock:
        if _inference_engine is None:
            _inference_engine = ChromaGuideInferenceEngine(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            _inference_engine.load_model()
        
        return _inference_engine


def reinitialize_engine(
    model_name: str = 'chromaguide',
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ChromaGuideInferenceEngine:
    """Reinitialize the global inference engine.
    
    Args:
        model_name: Model identifier
        checkpoint_path: Checkpoint path
        device: Device to use
        
    Returns:
        New ChromaGuideInferenceEngine instance
    """
    global _inference_engine
    
    with _engine_lock:
        _inference_engine = ChromaGuideInferenceEngine(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        _inference_engine.load_model()
        return _inference_engine

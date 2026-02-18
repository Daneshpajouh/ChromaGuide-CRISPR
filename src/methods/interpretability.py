"""
Interpretability Methods for ChromaGuide Model

Implements:
- Integrated Gradients for sequence attribution
- SHAP-based feature importance
- Attention weight visualization
- Saliency maps
- Feature interaction analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import logging
from matplotlib import pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class IntegratedGradients:
    """
    Integrated Gradients for interpreting neural network predictions.
    
    Computes attributions by integrating gradients along a straight line
    from a baseline to the input.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to extract gradients from (default: input)
        """
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device
        
    def _get_baseline(self, input_shape: Tuple) -> torch.Tensor:
        """Generate baseline (e.g., zero or random)."""
        baseline = torch.zeros(input_shape, device=self.device)
        return baseline
    
    def attribute(self, inputs: torch.Tensor, target_idx: Optional[int] = None,
                  n_steps: int = 50, baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Compute integrated gradients attribution.
        
        Args:
            inputs: Input tensor shape (batch_size, seq_len, features)
            target_idx: Target output index (None for regression)
            n_steps: Number of integration steps
            baseline: Custom baseline tensor
            
        Returns:
            Attributions shape (batch_size, seq_len, features)
        """
        if baseline is None:
            baseline = self._get_baseline(inputs.shape)
        
        inputs = inputs.to(self.device).requires_grad_(True)
        baseline = baseline.to(self.device)
        
        # Generate interpolated inputs along straight line from baseline to input
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        attributions = torch.zeros_like(inputs)
        
        for i, alpha in enumerate(alphas):
            # Interpolate
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            outputs = self.model(interpolated)
            
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Select target output
            if target_idx is not None:
                target_output = outputs[torch.arange(len(outputs)), target_idx]
            else:
                target_output = outputs.squeeze()
            
            # Backward pass
            self.model.zero_grad()
            target_output.sum().backward(retain_graph=True)
            
            # Accumulate gradients
            if isinstance(interpolated.grad, torch.Tensor):
                attributions += interpolated.grad.detach()
        
        # Average gradients and multiply by input difference
        attributions = attributions.mean(dim=0, keepdim=True) * (inputs - baseline)
        
        return attributions.detach().cpu().numpy()
    
    def explain_prediction(self, inputs: torch.Tensor, 
                          target_idx: Optional[int] = None) -> Dict:
        """
        Generate comprehensive explanation for a prediction.
        
        Returns:
            Dictionary with attributions and statistics
        """
        attributions = self.attribute(inputs, target_idx)
        
        # Aggregate attributions across feature dimension
        seq_level_attr = np.abs(attributions).sum(axis=2)
        
        # Feature importance
        feature_importance = np.abs(attributions).sum(axis=1).squeeze()
        
        return {
            'attributions': attributions,
            'seq_level': seq_level_attr,
            'feature_importance': feature_importance,
            'top_positions': np.argsort(-seq_level_attr.flatten())[:10],
            'top_features': np.argsort(-feature_importance)[:10],
        }


class SHAPInterpreter:
    """
    SHAP (SHapley Additive exPlanations) for feature importance.
    
    Computes Shapley values to quantify each feature's contribution to prediction.
    """
    
    def __init__(self, model, masker=None):
        """
        Args:
            model: PyTorch model
            masker: Function to mask/perturb inputs
        """
        self.model = model
        self.masker = masker
        self.device = next(model.parameters()).device
        self.background_data = None
        
    def set_background(self, background: torch.Tensor, sample_size: int = 100):
        """
        Set background data for SHAP baseline.
        
        Args:
            background: Background samples shape (n_samples, ...)
            sample_size: Number of samples to use
        """
        if len(background) > sample_size:
            indices = np.random.choice(len(background), sample_size, replace=False)
            background = background[indices]
        self.background_data = background.to(self.device)
        
    def _predict_on_masked(self, features: torch.Tensor, 
                          binary_mask: np.ndarray) -> np.ndarray:
        """
        Predict with features masked according to binary_mask.
        
        Args:
            features: Original features
            binary_mask: 1 = include, 0 = mask/replace with background
            
        Returns:
            Predictions
        """
        masked_features = features.clone()
        
        # Replace masked features with background mean
        for i, mask_val in enumerate(binary_mask):
            if mask_val == 0:
                if self.background_data is not None:
                    bg_mean = self.background_data.mean(dim=0)
                    masked_features[i] = bg_mean
                else:
                    masked_features[i] = 0
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(masked_features.to(self.device))
            if isinstance(output, tuple):
                output = output[0]
        
        return output.cpu().numpy()
    
    def shapley_values(self, sample: torch.Tensor, 
                      n_permutations: int = 100) -> np.ndarray:
        """
        Compute Shapley values for a sample.
        
        Args:
            sample: Single sample shape (1, ...)
            n_permutations: Number of permutations to average
            
        Returns:
            Shapley values shape same as sample
        """
        n_features = sample.shape[1] if len(sample.shape) > 1 else 1
        shapley_values = np.zeros(n_features)
        
        for _ in range(n_permutations):
            # Random permutation
            perm = np.random.permutation(n_features)
            
            for i, feature_idx in enumerate(perm):
                # Prediction with feature
                with_mask = np.ones(n_features)
                with_mask[feature_idx] = 1
                pred_with = self._predict_on_masked(sample, with_mask)
                
                # Prediction without feature
                without_mask = np.ones(n_features)
                without_mask[feature_idx] = 0
                pred_without = self._predict_on_masked(sample, without_mask)
                
                # Contribution
                shapley_values[feature_idx] += (pred_with - pred_without).mean()
        
        return shapley_values / n_permutations
    
    def explain_batch(self, samples: torch.Tensor, 
                     n_permutations: int = 50) -> Dict:
        """Explain multiple samples."""
        batch_shapley = []
        
        for i in range(len(samples)):
            shapley = self.shapley_values(samples[i:i+1], n_permutations)
            batch_shapley.append(shapley)
        
        batch_shapley = np.array(batch_shapley)
        
        return {
            'shapley_values': batch_shapley,
            'feature_importance': np.abs(batch_shapley).mean(axis=0),
            'mean_abs_shapley': np.abs(batch_shapley).mean(),
        }


class AttentionVisualizer:
    """Visualize attention weights from transformer models."""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        for module in self.model.modules():
            if hasattr(module, 'attention'):
                def hook_fn(module, input, output):
                    # Store attention weights
                    if isinstance(output, tuple) and len(output) > 0:
                        attn_weights = output[0]
                        if isinstance(attn_weights, torch.Tensor):
                            self.attention_weights['last'] = attn_weights.detach()
                
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def get_attention_map(self, inputs: torch.Tensor) -> Optional[np.ndarray]:
        """
        Extract attention map for inputs.
        
        Args:
            inputs: Input tensor shape (batch, seq_len, features)
            
        Returns:
            Attention map shape (batch, seq_len, seq_len)
        """
        self.model.eval()
        self.attention_weights = {}
        
        with torch.no_grad():
            _ = self.model(inputs)
        
        if 'last' in self.attention_weights:
            attn = self.attention_weights['last']
            # Average over heads if multi-head
            if len(attn.shape) == 4:
                attn = attn.mean(dim=1)
            return attn.cpu().numpy()
        
        return None
    
    def visualize_attention(self, inputs: torch.Tensor, seq_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None):
        """
        Visualize attention heatmap.
        
        Args:
            inputs: Input tensor
            seq_names: Names for sequence positions
            save_path: Path to save figure
        """
        attention = self.get_attention_map(inputs)
        
        if attention is None:
            logger.warning("No attention weights captured")
            return
        
        # Take first sample in batch
        attention = attention[0]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(attention, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention'})
        
        ax.set_title('Attention Weights Heatmap')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        if seq_names:
            ax.set_xticklabels(seq_names, rotation=45)
            ax.set_yticklabels(seq_names)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        plt.close()
    
    def cleanup(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()


class SaliencyMap:
    """Generate saliency maps showing which input regions affect predictions."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def compute_saliency(self, inputs: torch.Tensor, 
                        target_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute saliency map using input gradients.
        
        Args:
            inputs: Input tensor shape (batch, seq_len, features)
            target_idx: Target output index
            
        Returns:
            Saliency map shape (batch, seq_len)
        """
        inputs = inputs.to(self.device).requires_grad_(True)
        
        self.model.eval()
        outputs = self.model(inputs)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Select target output
        if target_idx is not None:
            target_output = outputs[torch.arange(len(outputs)), target_idx]
        else:
            target_output = outputs.squeeze()
        
        # Backward pass
        self.model.zero_grad()
        target_output.sum().backward(retain_graph=True)
        
        # Saliency: max absolute gradient across feature dimension
        saliency = torch.abs(inputs.grad).max(dim=2)[0]
        
        return saliency.detach().cpu().numpy()
    
    def visualize_saliency(self, inputs: torch.Tensor, seq_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
        """Visualize saliency map."""
        saliency = self.compute_saliency(inputs)
        
        # Plot first sample
        saliency = saliency[0]
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(saliency)), saliency)
        
        ax.set_title('Saliency Map (Gradient-based)')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Gradient Magnitude')
        
        if seq_names:
            ax.set_xticks(range(len(seq_names)))
            ax.set_xticklabels(seq_names, rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saliency map saved to {save_path}")
        
        plt.close()


class FeatureInteractionAnalyzer:
    """Analyze pairwise feature interactions in model predictions."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def compute_interaction_matrix(self, inputs: torch.Tensor,
                                  n_features: int) -> np.ndarray:
        """
        Compute pairwise feature interaction scores.
        
        Uses second-order partial derivatives to measure interaction strength.
        """
        interaction_matrix = np.zeros((n_features, n_features))
        
        inputs = inputs.to(self.device).requires_grad_(True)
        
        for i in range(n_features):
            for j in range(i, n_features):
                self.model.eval()
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # First gradient w.r.t. feature i
                grad_i = torch.autograd.grad(outputs.sum(), inputs, 
                                            retain_graph=True, create_graph=True)[0]
                
                # Second gradient w.r.t. feature j
                grad_ij = torch.autograd.grad(grad_i[:, :, i].sum(), inputs,
                                             create_graph=False)[0]
                
                interaction_score = torch.abs(grad_ij[:, :, j]).mean().item()
                interaction_matrix[i, j] = interaction_score
                interaction_matrix[j, i] = interaction_score
        
        return interaction_matrix
    
    def visualize_interactions(self, interaction_matrix: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
        """Visualize feature interaction heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(interaction_matrix, cmap='RdYlBu_r', ax=ax, 
                   cbar_kws={'label': 'Interaction Strength'})
        
        ax.set_title('Feature Interaction Matrix')
        
        if feature_names:
            ax.set_xticklabels(feature_names, rotation=45)
            ax.set_yticklabels(feature_names)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Interaction matrix saved to {save_path}")
        
        plt.close()


class InterpretabilityReporter:
    """Generate comprehensive interpretability reports."""
    
    @staticmethod
    def generate_report(sample_idx: int,
                       integrated_grads: Dict,
                       shap_values: Dict,
                       attention: Optional[np.ndarray] = None,
                       saliency: Optional[np.ndarray] = None) -> str:
        """Generate interpretability analysis report."""
        
        report = f"""
INTERPRETABILITY ANALYSIS REPORT
=================================
Sample Index: {sample_idx}

INTEGRATED GRADIENTS
--------------------
Top Attribution Positions (by absolute value):
"""
        
        if 'top_positions' in integrated_grads:
            for i, pos in enumerate(integrated_grads['top_positions'][:5], 1):
                report += f"  {i}. Position {pos}\n"
        
        if 'top_features' in integrated_grads:
            report += "\nTop Contributing Features:\n"
            for i, feat in enumerate(integrated_grads['top_features'][:5], 1):
                report += f"  {i}. Feature {feat}\n"
        
        report += "\nSHAP VALUES\n-----------\n"
        if 'feature_importance' in shap_values:
            avg_importance = shap_values['feature_importance']
            report += f"Mean Absolute SHAP: {shap_values['mean_abs_shapley']:.6f}\n"
            report += f"Feature Importance Range: [{avg_importance.min():.6f}, {avg_importance.max():.6f}]\n"
        
        if attention is not None:
            report += f"\nATTENTION\n---------\n"
            report += f"Attention Pattern Shape: {attention.shape}\n"
            report += f"Mean Attention: {attention.mean():.6f}\n"
            report += f"Max Attention: {attention.max():.6f}\n"
        
        if saliency is not None:
            report += f"\nSALIENCY\n--------\n"
            report += f"Mean Gradient: {saliency.mean():.6f}\n"
            report += f"Peak Gradient: {saliency.max():.6f}\n"
        
        return report

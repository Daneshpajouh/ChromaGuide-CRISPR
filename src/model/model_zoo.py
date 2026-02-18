"""Model zoo: Auto-registration of all ChromaGuide models.

This module imports all model implementations and registers them with the
ModelRegistry for use with the ModelFactory.

Models are organized by category:
  - CRISPRO variants: Core SOTA models
  - ChromaGuide: Multi-modal on-target prediction
  - DNABERT-Mamba: Foundation + SSM hybrids
  - DeepMENS: Ensemble architecture
  - Generative: VAE and diffusion models
  - Uncertainty: Conformal and probabilistic variants
"""

import logging
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


def register_all_models() -> None:
    """Register all available model architectures.
    
    Call this function at startup to populate the registry.
    """
    
    # =========================================================================
    # CRISPRO VARIANTS - Core State Space Models
    # =========================================================================
    try:
        from .crispro import CRISPROModel
        
        ModelRegistry.register(
            'crispro',
            'CRISPR efficiency prediction with Mamba-2 encoding'
        )(CRISPROModel)
        
        ModelRegistry.register_default_config('crispro', {
            'd_model': 256,
            'n_layers': 4,
            'n_modalities': 33,
            'vocab_size': 20,
            'use_causal': False,
            'use_quantum': False,
            'use_topo': False,
        })
        logger.info("Registered: crispro")
    except ImportError as e:
        logger.warning(f"Could not register 'crispro': {e}")
    
    # CRISPRO variants with enhancements
    try:
        from .crispro_mamba_x import CRISPROMambaX
        
        ModelRegistry.register(
            'crispro_mamba_x',
            'CRISPRO with quantum tunneling + topological regularization'
        )(CRISPROMambaX)
        
        ModelRegistry.register_default_config('crispro_mamba_x', {
            'd_model': 256,
            'n_layers': 4,
            'use_quantum': True,
            'use_topo': True,
            'use_causal': True,
        })
        logger.info("Registered: crispro_mamba_x")
    except ImportError as e:
        logger.warning(f"Could not register 'crispro_mamba_x': {e}")
    
    # =========================================================================
    # CHROMAGUIDE - Multi-Modal On-Target Prediction
    # =========================================================================
    try:
        from ..chromaguide.chromaguide_model import ChromaGuideModel
        
        ModelRegistry.register(
            'chromaguide',
            'Multi-modal on-target efficacy with epigenomic fusion (Beta regression)'
        )(ChromaGuideModel)
        
        ModelRegistry.register_default_config('chromaguide', {
            'encoder_type': 'mamba',  # 'cnn_gru' or 'mamba'
            'd_model': 256,
            'seq_len': 23,
            'num_epi_tracks': 4,
            'num_epi_bins': 100,
            'use_epigenomics': True,
            'use_gate_fusion': False,
            'use_mi_regularizer': False,
            'mi_lambda': 0.01,
            'dropout': 0.2,
            'mamba_layers': 4,
            'mamba_d_state': 16,
            'mamba_d_conv': 4,
            'mamba_expand': 2,
        })
        logger.info("Registered: chromaguide")
    except ImportError as e:
        logger.warning(f"Could not register 'chromaguide': {e}")
    
    # =========================================================================
    # DNABERT-MAMBA HYBRIDS - Foundation Model Integration
    # =========================================================================
    try:
        from .dnabert_mamba import DNABERTMamba
        
        ModelRegistry.register(
            'dnabert_mamba',
            'DNABERT foundation model + Mamba adapter with multi-head outputs'
        )(DNABERTMamba)
        
        ModelRegistry.register_default_config('dnabert_mamba', {
            'dnabert_name': 'zhihan1996/DNABERT-2-117M',
            'adapter_kind': 'linear',
            'adapter_out_dim': 256,
            'mamba_cfg': {
                'd_model': 256,
                'n_layers': 4,
            },
        })
        logger.info("Registered: dnabert_mamba")
    except ImportError as e:
        logger.warning(f"Could not register 'dnabert_mamba': {e}")
    
    # =========================================================================
    # DEEPMENS - Multi-Branch Ensemble Architecture
    # =========================================================================
    try:
        from .deepmens import DeepMEnsExact
        
        ModelRegistry.register(
            'deepmens',
            'DeepMENS multi-scale CNN ensemble (sequence + shape + position branches)'
        )(DeepMEnsExact)
        
        ModelRegistry.register_default_config('deepmens', {
            'seq_len': 23,
            'num_shape_features': 4,
            'dropout': 0.3,
        })
        logger.info("Registered: deepmens")
    except ImportError as e:
        logger.warning(f"Could not register 'deepmens': {e}")
    
    try:
        from .ensemble import DeepMEnsEnsemble
        
        ModelRegistry.register(
            'deepmens_ensemble',
            'DeepMENS ensemble wrapper (averages 5 models with uncertainty)'
        )(DeepMEnsEnsemble)
        
        logger.info("Registered: deepmens_ensemble")
    except ImportError as e:
        logger.warning(f"Could not register 'deepmens_ensemble': {e}")
    
    # =========================================================================
    # GENERATIVE MODELS - VAE and Diffusion
    # =========================================================================
    try:
        from .rnagenesis_vae import RNAGenesisVAE
        
        ModelRegistry.register(
            'rnagenesis_vae',
            'VAE for synthetic guide RNA generation'
        )(RNAGenesisVAE)
        
        ModelRegistry.register_default_config('rnagenesis_vae', {
            'seq_len': 23,
            'latent_dim': 64,
            'hidden_dim': 256,
        })
        logger.info("Registered: rnagenesis_vae")
    except ImportError as e:
        logger.warning(f"Could not register 'rnagenesis_vae': {e}")
    
    try:
        from .rnagenesis_diffusion import RNAGenesisDiffusion
        
        ModelRegistry.register(
            'rnagenesis_diffusion',
            'Diffusion model for synthetic guide RNA generation'
        )(RNAGenesisDiffusion)
        
        ModelRegistry.register_default_config('rnagenesis_diffusion', {
            'seq_len': 23,
            'hidden_dim': 256,
            'noise_steps': 1000,
        })
        logger.info("Registered: rnagenesis_diffusion")
    except ImportError as e:
        logger.warning(f"Could not register 'rnagenesis_diffusion': {e}")
    
    # =========================================================================
    # UNCERTAINTY & CONFORMAL
    # =========================================================================
    try:
        from .conformal import ConformalPredictor
        
        ModelRegistry.register(
            'conformal',
            'Wrapper for conformal prediction (coverage guarantees)'
        )(ConformalPredictor)
        
        logger.info("Registered: conformal")
    except ImportError as e:
        logger.warning(f"Could not register 'conformal': {e}")
    
    # =========================================================================
    # OFF-TARGET & RAG MODELS
    # =========================================================================
    try:
        from .off_target import OffTargetPredictor
        
        ModelRegistry.register(
            'off_target',
            'CNN-based off-target risk prediction (alignment pairs)'
        )(OffTargetPredictor)
        
        ModelRegistry.register_default_config('off_target', {
            'seq_len': 23,
            'hidden_dim': 128,
            'num_conv_layers': 3,
        })
        logger.info("Registered: off_target")
    except ImportError as e:
        logger.warning(f"Could not register 'off_target': {e}")
    
    try:
        from .ot_rag import OTRetrieverAugmentedGenerator
        
        ModelRegistry.register(
            'ot_rag',
            'Retrieval-augmented generation for off-target prediction'
        )(OTRetrieverAugmentedGenerator)
        
        logger.info("Registered: ot_rag")
    except ImportError as e:
        logger.warning(f"Could not register 'ot_rag': {e}")
    
    # =========================================================================
    # SPECIALIZED VARIANTS
    # =========================================================================
    try:
        from .mamba_deepmens import MambaDeepMENS
        
        ModelRegistry.register(
            'mamba_deepmens',
            'Integration of Mamba with DeepMENS multi-branch architecture'
        )(MambaDeepMENS)
        
        logger.info("Registered: mamba_deepmens")
    except ImportError as e:
        logger.warning(f"Could not register 'mamba_deepmens': {e}")
    
    # LogSummary
    models = ModelRegistry.list_models()
    logger.info(f"=" * 70)
    logger.info(f"Model Zoo Loaded: {len(models)} models registered")
    for name in models:
        desc = ModelRegistry.get_description(name)
        logger.info(f"  • {name:25s} - {desc}")
    logger.info(f"=" * 70)


# Auto-register on module import
register_all_models()

"""ChromaGuide: Multi-Modal Deep Learning Framework for CRISPR-Cas9 Guide RNA Design."""

from .sequence_encoder import SequenceEncoder, CNNGRUEncoder, MambaSequenceEncoder
from .epigenomic_encoder import EpigenomicEncoder
from .fusion import ChromaGuideFusion
from .prediction_head import BetaRegressionHead
from .off_target import OffTargetScorer, CandidateFinder
from .design_score import IntegratedDesignScore
from .chromaguide_model import ChromaGuideModel

__version__ = '0.1.0'
__all__ = [
    'SequenceEncoder', 'CNNGRUEncoder', 'MambaSequenceEncoder',
    'EpigenomicEncoder', 'ChromaGuideFusion', 'BetaRegressionHead',
    'OffTargetScorer', 'CandidateFinder', 'IntegratedDesignScore',
    'ChromaGuideModel',
]

"""
AutoFeature: Intelligent Feature Engineering for Tabular ML
"""

from .engineer import AutoFeatureEngineer
from .selector import TargetAwareSelector
from .encoders import CyclicalEncoder, SmartCategoricalEncoder
from .detector import LeakageDetector
from .pipeline import AutoFeaturePipeline

__version__ = "0.1.0"
__author__ = "Sufyaan"

__all__ = [
    "AutoFeatureEngineer",
    "TargetAwareSelector",
    "CyclicalEncoder",
    "SmartCategoricalEncoder",
    "LeakageDetector",
    "AutoFeaturePipeline",
]

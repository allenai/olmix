"""Model configuration module for olmix experiments."""

from olmix.model.aliases import ModelTrainConfig
from olmix.model.transformer import TransformerConfigBuilder

__all__ = [
    "ModelTrainConfig",
    "TransformerConfigBuilder",
]

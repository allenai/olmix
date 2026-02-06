"""Model configuration for olmix experiments."""

from dataclasses import dataclass

import olmo_core.train.train_module as tm
from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import TrainerConfig


@dataclass
class ModelTrainConfig(Config):
    """Complete training configuration for a transformer model."""

    model: TransformerConfig
    train_module: tm.TransformerTrainModuleConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536

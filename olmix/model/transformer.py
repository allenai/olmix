"""Transformer configuration builder for olmix experiments."""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import olmo_core.train.train_module as tm
from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    WSDS,
    OptimGroupOverride,
    Scheduler,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy

from olmix.aliases import SourceInstance, TrainType
from olmix.model.aliases import ModelTrainConfig
from olmix.utils.cloud import expand_cloud_globs

logger = logging.getLogger(__name__)

# Constants for scaling law calculations
BATCH_DIVISOR = 32
SAVE_INTERVAL = 1000
SEQUENCE_LENGTH = 8192  # Fixed sequence length for Chinchilla strategy
TOKENS_PER_PARAM = 20  # Chinchilla optimal tokens per parameter

# Direct factory mappings to olmo-core
TOKENIZERS: dict[str, Callable[[], TokenizerConfig]] = {
    "dolma2": TokenizerConfig.dolma2,
    "gpt_neox": TokenizerConfig.gpt_neox_olmo_dolma_v1_5,
}


def _get_model_factory(tokenizer: TokenizerConfig) -> dict[str, Callable[[], TransformerConfig]]:
    """Get model factories with the given tokenizer's vocab size."""
    vocab_size = tokenizer.padded_vocab_size()
    return {
        # OLMo2 models
        "olmo2_1m": lambda: TransformerConfig.olmo2_1M(vocab_size=vocab_size),
        "olmo2_30m": lambda: TransformerConfig.olmo2_30M(vocab_size=vocab_size),
        "olmo2_60m": lambda: TransformerConfig.olmo2_60M(vocab_size=vocab_size),
        "olmo2_190m": lambda: TransformerConfig.olmo2_190M(vocab_size=vocab_size),
        "olmo2_1b": lambda: TransformerConfig.olmo2_1B_v2(vocab_size=vocab_size),
        "olmo2_7b": lambda: TransformerConfig.olmo2_7B_v2(vocab_size=vocab_size),
        # OLMo3 models
        "olmo3_1m": lambda: TransformerConfig.olmo3_1M(vocab_size=vocab_size),
        "olmo3_14m": lambda: TransformerConfig.olmo3_14M(vocab_size=vocab_size),
        "olmo3_30m": lambda: TransformerConfig.olmo3_30M(vocab_size=vocab_size),
        "olmo3_60m": lambda: TransformerConfig.olmo3_60M(vocab_size=vocab_size),
        "olmo3_100m": lambda: TransformerConfig.olmo3_100M(vocab_size=vocab_size),
        "olmo3_190m": lambda: TransformerConfig.olmo3_190M(vocab_size=vocab_size),
        "olmo3_370m": lambda: TransformerConfig.olmo3_370M(vocab_size=vocab_size),
        "olmo3_600m": lambda: TransformerConfig.olmo3_600M(vocab_size=vocab_size),
        "olmo3_760m": lambda: TransformerConfig.olmo3_760M(vocab_size=vocab_size),
        "olmo3_1b": lambda: TransformerConfig.olmo3_1B(vocab_size=vocab_size),
        "olmo3_3b": lambda: TransformerConfig.olmo3_3B(vocab_size=vocab_size),
        "olmo3_7b": lambda: TransformerConfig.olmo3_7B(vocab_size=vocab_size),
        "olmo3_13b": lambda: TransformerConfig.olmo3_13B(vocab_size=vocab_size),
        "olmo3_32b": lambda: TransformerConfig.olmo3_32B(vocab_size=vocab_size),
    }


@dataclass
class TransformerConfigBuilder:
    """
    A builder class for configuring and creating a transformer model training configuration.

    Uses Chinchilla-based scaling laws for batch size, learning rate, and duration.
    Uses WSDS (Warmup-Stable-Decay-Stable) scheduler with multi-period checkpoints.

    Attributes:
        run_name: The name of the run.
        sources: A list of source instances.
        chinchilla_multiple: Multiplier for Chinchilla optimal tokens (trains for 20 * params * N).
        transformer_config: The model configuration (TransformerConfig from olmo-core).
        group_id: The group ID for the run.
        cluster: The cluster name.
        beaker_user: The Beaker user name.
        s3: Whether to use S3 for storage.
        seed: The random seed for reproducibility.
        tokenizer: The tokenizer configuration.
        dtype: The data type for the dataset.
        weka: Whether to use Weka buckets.
        train_type: The training type.
        load_path: The path to load a pre-trained model.
        profile: Whether to enable profiling.
    """

    run_name: str
    sources: list[SourceInstance]
    chinchilla_multiple: float
    transformer_config: TransformerConfig
    group_id: str
    cluster: str
    beaker_user: str
    s3: bool
    seed: int
    tokenizer: TokenizerConfig
    dtype: str
    weka: bool
    device_batch_size: int
    load_path: str | None = None
    profile: bool = False
    train_type: TrainType = TrainType.pretrain

    def __init__(
        self,
        run_name: str,
        sources: list[SourceInstance],
        chinchilla_multiple: float,
        group_id: str,
        cluster: str,
        beaker_user: str,
        tokenizer: str,
        dtype: str,
        model_identifier: str,
        weka: bool,
        device_batch_size: int,
        train_type: TrainType = TrainType.pretrain,
        load_path: str | None = None,
        seed: int = 42,
        s3: bool = True,
        profile: bool = False,
        global_batch_size: int | None = None,
    ):
        self.run_name = run_name
        self.sources = sources
        self.chinchilla_multiple = chinchilla_multiple
        self.sequence_length = SEQUENCE_LENGTH  # Fixed at 8192
        self.group_id = group_id
        self.seed = seed
        self.beaker_user = beaker_user
        self.profile = profile
        self.s3 = s3
        self.train_type = train_type
        self.load_path = load_path
        self.device_batch_size = device_batch_size
        self.global_batch_size = global_batch_size

        # Use olmo-core directly for tokenizer
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"Unknown tokenizer: {tokenizer}. Available: {list(TOKENIZERS.keys())}")
        self.tokenizer = TOKENIZERS[tokenizer]()

        # Use olmo-core directly for model config
        models = _get_model_factory(self.tokenizer)
        if model_identifier not in models:
            raise ValueError(f"Unknown model: {model_identifier}. Available: {list(models.keys())}")
        self.transformer_config = models[model_identifier]()

        self.data_dir: str = "s3://ai2-llm"
        self.dataset_dtype = NumpyDatasetDType[dtype]
        self.root_dir = f"/tmp/{self.run_name}"
        self.cluster = cluster
        self.weka = weka

        # Default will always be s3 for checkpoints, and we override if Augusta or AUS+Weka
        self.checkpoint_dir = f"{self.data_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"

        self._setup_dirs()

    def _setup_dirs(self) -> None:
        """Setup checkpoint directory based on cluster configuration."""
        if any(substring in self.cluster for substring in ["augusta"]):
            self.root_dir = "gs://ai2-llm"
            self.checkpoint_dir = f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
            # NOTE: work_dir must be a local path, not a url
            self.work_dir = f"/tmp/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"
        elif (
            any(substring in self.cluster for substring in ["jupiter", "saturn", "ceres", "neptune", "titan"])
            and self.weka
        ):
            logger.info("Using Weka bucket as root dir")
            self.root_dir = "/weka/oe-training-default/ai2-llm"
            self.checkpoint_dir = f"{self.root_dir}/checkpoints/{self.beaker_user.lower()}/{self.run_name}"
            self.work_dir = f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"
        else:
            self.work_dir = f"{self.root_dir}/{self.beaker_user.lower()}/{self.run_name}/dataset-cache"

    def get_warmup_tokens(self, num_params: int) -> int:
        """Returns the number of warmup tokens.

        Uses 1 token per parameter, but capped at 5% of total duration to ensure
        warmup doesn't exceed total training time for small chinchilla_multiple.
        """
        default_warmup = num_params
        max_tokens = self.get_duration(num_params)
        max_warmup = int(max_tokens * 0.05)  # Cap at 5% of total duration
        return min(default_warmup, max_warmup)

    def get_batch_size(self, num_params: int) -> int:
        """
        Returns the global batch size based on model parameters using Chinchilla formula.

        Formula: 2048 * 160 * (N / 108M)^(2/3), rounded to next power of 2.
        """
        batch_size = round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))
        return self.next_power_of_2(batch_size)

    def next_power_of_2(self, x: int) -> int:
        """Returns the next power of 2 greater than or equal to x."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def get_lr(self, num_params: int) -> float:
        """
        Returns the learning rate using Chinchilla formula with halving for stability.

        Formula: 0.0047 * (N / 108M)^(-1/3) / 2
        """
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        return lr / 2.0

    def get_duration(self, num_params: int) -> int:
        """
        Returns the total training duration in tokens using Chinchilla formula.

        Formula: TOKENS_PER_PARAM * num_params * chinchilla_multiple
        """
        return int(TOKENS_PER_PARAM * num_params * self.chinchilla_multiple)

    def get_scheduler(self, num_params: int, batch_size: int) -> Scheduler:
        """
        Returns WSDS scheduler with Chinchilla-based periods.

        Periods are generated at 0.5xC, 1xC, 2xC, ... up to chinchilla_multiple.
        For small chinchilla_multiple (< 0.5), uses a single period for the full duration.
        """
        warmup = self.get_warmup_tokens(num_params)
        max_tokens = self.get_duration(num_params)

        # Generate periods: 0.5xC, 1xC, 2xC, ... up to target
        periods: list[float] = []
        p = 0.5
        while p <= self.chinchilla_multiple:
            periods.append(p)
            p *= 2

        # If no periods (chinchilla_multiple < 0.5), use single period for full duration
        if not periods:
            # Round to batch boundary
            period_length = batch_size * max(1, round(max_tokens / batch_size))
            return WSDS(
                units=SchedulerUnits.tokens,
                warmup=warmup,
                decay_fraction=0.1,
                period_lengths=[period_length],
            )

        # Calculate period lengths (tokens between checkpoints)
        period_lengths: list[int] = []
        for i, c in enumerate(periods):
            tokens_at_c = int(TOKENS_PER_PARAM * num_params * c)
            # Round to nearest batch boundary
            tokens_at_c = batch_size * round(tokens_at_c / batch_size)
            if i == 0:
                period_lengths.append(tokens_at_c)
            else:
                prev_c = periods[i - 1]
                prev_tokens = int(TOKENS_PER_PARAM * num_params * prev_c)
                prev_tokens = batch_size * round(prev_tokens / batch_size)
                period_lengths.append(tokens_at_c - prev_tokens)

        return WSDS(
            units=SchedulerUnits.tokens,
            warmup=warmup,
            decay_fraction=0.1,
            period_lengths=period_lengths,
        )

    def build_callbacks(self) -> dict[str, Callback]:
        """Builds and returns a dictionary of callbacks for the trainer."""
        return {
            "gpu_monitor": GPUMemoryMonitorCallback(),
            "config_saver": ConfigSaverCallback(),
            "profiler": ProfilerCallback(enabled=self.profile),
            "checkpointer": CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=250,
                save_async=True,
            ),
            "wandb": WandBCallback(
                name=self.run_name.strip(),
                project="olmix",
                group=self.group_id.strip(),
                enabled=True,
            ),
        }

    def build(self) -> ModelTrainConfig:
        """Builds and returns the model training configuration."""
        tokenizer = self.tokenizer
        model = self.transformer_config
        num_params = model.num_non_embedding_params

        global_batch_size = (
            self.global_batch_size if self.global_batch_size is not None else self.get_batch_size(num_params)
        )
        learning_rate = self.get_lr(num_params)
        max_tokens = self.get_duration(num_params)

        # Adaptive beta2: use 0.95 for large batches, 0.99 for smaller batches
        batch_size_in_tokens = global_batch_size * self.sequence_length
        beta2 = 0.95 if batch_size_in_tokens >= 524_288 else 0.99

        # Build source mixture (inlined from MixtureBuilder)
        source_configs = SourceMixtureList(sources=[])
        for source in self.sources:
            globs = [p for p in source.paths if "*" in p]
            paths = [p for p in source.paths if "*" not in p]
            source_configs.sources.append(
                SourceMixtureConfig(
                    source_name=source.name,
                    paths=paths + expand_cloud_globs(globs),
                    target_ratio=source.ratio,
                    max_repetition_ratio=source.repetition_factor,
                )
            )

        mixture_config = SourceMixtureDatasetConfig(
            source_list=source_configs,
            requested_tokens=max_tokens,
            global_batch_size=batch_size_in_tokens,
            seed=self.seed,
            processes=min(os.cpu_count() or 1, 16),
        )

        dataset_config = NumpyFSLDatasetConfig(
            source_mixture_config=mixture_config,
            sequence_length=self.sequence_length,
            tokenizer=tokenizer,
            work_dir=self.work_dir,
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=batch_size_in_tokens,
            work_dir=self.work_dir,
            seed=self.seed,
            num_workers=4,
        )

        train_module_config = tm.TransformerTrainModuleConfig(
            rank_microbatch_size=self.device_batch_size * self.sequence_length,
            max_sequence_length=self.sequence_length,
            optim=SkipStepAdamWConfig(
                lr=learning_rate,
                weight_decay=0.1,
                betas=(0.9, beta2),
                group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))],
            ),
            compile_model=True,
            dp_config=tm.TransformerDataParallelConfig(
                name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
            ),
            float8_config=Float8Config(enabled=False),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=self.get_scheduler(num_params, global_batch_size),
        )

        trainer_config = TrainerConfig(
            save_folder=self.checkpoint_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            load_path=self.load_path,
            # We fail fast if an existing if we expect a checkpoint for annealing and one is not found.
            load_strategy=(LoadStrategy.always if self.train_type == TrainType.anneal else LoadStrategy.if_available),
            max_duration=Duration.tokens(max_tokens),
        )

        for callback_name, callback in self.build_callbacks().items():
            trainer_config.callbacks[callback_name] = callback

        return ModelTrainConfig(
            model=model,
            dataset=dataset_config,
            data_loader=data_loader_config,
            trainer=trainer_config,
            train_module=train_module_config,
        )

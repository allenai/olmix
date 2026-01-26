# Plan 003: Remove Redundant olmix Abstractions in Favor of Direct olmo-core Usage

## Overview

Simplify olmix by removing thin wrapper abstractions and using olmo-core APIs directly. This reduces code complexity and maintenance burden while preserving olmix-specific functionality.

**Goals:**
1. Remove `SupportedModels`, `SupportedTokenizers`, `TOKENIZER` constants
2. Remove `MixtureBuilder` class (inline + extract glob utility)
3. Simplify `TransformerConfigBuilder` to use olmo-core directly
4. Keep `ModelTrainConfig` (valuable composition)
5. Keep scaling law logic (olmix-specific)

---

## Analysis Summary

### Abstractions to REMOVE

| Component | Location | Reason |
|-----------|----------|--------|
| `TOKENIZER` | `olmix/model/aliases.py` | Just `TokenizerConfig.dolma2()` - inline it |
| `SupportedTokenizers` | `olmix/model/aliases.py` | Thin enum wrapper - use olmo-core directly |
| `SupportedModels` | `olmix/model/aliases.py` | Thin wrapper around `TransformerConfig.olmo2_*()` |
| `get_model_config()` | `olmix/model/aliases.py` | Legacy name mapping - remove with migration |
| `MixtureBuilder` | `olmix/data/dataset.py` | Inline into `TransformerConfigBuilder`, extract glob utility |

### Abstractions to KEEP

| Component | Location | Reason |
|-----------|----------|--------|
| `ModelTrainConfig` | `olmix/model/aliases.py` | Useful composition of 5 configs |
| `TransformerConfigBuilder` | `olmix/model/transformer.py` | Contains scaling laws, cluster setup |
| Scaling law functions | `olmix/model/transformer.py` | olmix-specific logic |
| `expand_globs()` | Extract to utility | S3/cloud glob expansion needed |

---

## Implementation Plan

### Phase 1: Extract Glob Utility

Create `olmix/utils/cloud.py`:

```python
"""Cloud storage utilities."""

from typing import List
from urllib.parse import urlparse

import s3fs
from olmo_core.io import is_url


def expand_cloud_globs(paths: List[str], fs: s3fs.S3FileSystem | None = None) -> List[str]:
    """Expand glob patterns in cloud storage paths.

    Args:
        paths: List of paths, some may contain glob patterns (*)
        fs: Optional S3FileSystem instance (created if not provided)

    Returns:
        List of expanded paths with globs resolved
    """
    if fs is None:
        fs = s3fs.S3FileSystem()

    results = []
    for path in paths:
        if "*" not in path:
            results.append(path)
            continue

        if not is_url(path):
            raise NotImplementedError("Glob expansion only supported for URLs")

        parsed = urlparse(str(path))
        if parsed.scheme in ("s3", "r2", "weka"):
            results.extend([f"s3://{obj}" for obj in fs.glob(path)])
        elif parsed.scheme == "gs":
            raise NotImplementedError("'gs' glob expansion not supported")
        else:
            raise NotImplementedError(f"Glob expansion not supported for '{parsed.scheme}'")

    return results
```

### Phase 2: Simplify `olmix/model/aliases.py`

**Before:** 111 lines with `SupportedModels`, `SupportedTokenizers`, `TOKENIZER`, `get_model_config()`

**After:** ~30 lines - just `ModelTrainConfig`

```python
"""Model configuration for olmix experiments."""

from dataclasses import dataclass

from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import TrainerConfig
import olmo_core.train.train_module as tm


@dataclass
class ModelTrainConfig(Config):
    """Complete training configuration for a transformer model."""

    model: TransformerConfig
    train_module: tm.TransformerTrainModuleConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536
```

### Phase 3: Update `olmix/model/transformer.py`

Replace abstraction usage with direct olmo-core calls:

```python
# Before
from olmix.model.aliases import (
    ModelTrainConfig,
    SupportedTokenizers,
    get_model_config,
)

self.transformer_config = get_model_config(model_identifier)
self.tokenizer = SupportedTokenizers[tokenizer].value

# After
from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from olmix.model.aliases import ModelTrainConfig

# Direct factory calls
TOKENIZERS = {
    "dolma2": TokenizerConfig.dolma2,
    "gpt_neox": TokenizerConfig.gpt_neox_olmo_dolma_v1_5,
}

MODELS = {
    "olmo2_1m": TransformerConfig.olmo2_1M,
    "olmo2_30m": TransformerConfig.olmo2_30M,
    "olmo2_60m": TransformerConfig.olmo2_60M,
    "olmo2_190m": TransformerConfig.olmo2_190M,
    "olmo2_1b": TransformerConfig.olmo2_1B_v2,
    "olmo2_7b": TransformerConfig.olmo2_7B_v2,
}

# In __init__:
self.tokenizer = TOKENIZERS[tokenizer]()
self.transformer_config = MODELS[model_identifier](
    vocab_size=self.tokenizer.padded_vocab_size()
)
```

Inline `MixtureBuilder` logic into `build()` method:

```python
from olmix.utils.cloud import expand_cloud_globs
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)

def build(self) -> ModelTrainConfig:
    # Inline mixture building (was MixtureBuilder)
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
        requested_tokens=self.max_tokens,
        global_batch_size=global_batch_size * self.sequence_length,
        seed=self.seed,
        processes=min(os.cpu_count() or 1, 16),
    )
    # ... rest of build()
```

### Phase 4: Remove `olmix/data/dataset.py`

Delete `MixtureBuilder` class entirely. The file can be removed if it only contains `MixtureBuilder`.

### Phase 5: Update Config Files

Update `configs/examples/*.yaml` to use model identifiers that match olmo-core factory names:

```yaml
# configs/examples/test_olmo2_30m.yaml
proxy_model_id: olmo2_30m  # Maps to TransformerConfig.olmo2_30M
tokenizer: dolma2          # Maps to TokenizerConfig.dolma2
```

### Phase 6: Update Tests

Update `tests/test_imports.py`:

```python
def test_model_module_imports(self):
    """Test model module imports."""
    from olmix.model.aliases import ModelTrainConfig
    # Removed: SupportedModels, SupportedTokenizers, get_model_config, TOKENIZER

    assert ModelTrainConfig is not None

def test_direct_olmo_core_usage(self):
    """Test that olmo-core can be used directly."""
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.data import TokenizerConfig

    tokenizer = TokenizerConfig.dolma2()
    model = TransformerConfig.olmo2_30M(vocab_size=tokenizer.padded_vocab_size())

    assert model.d_model == 256
    assert tokenizer.vocab_size == 100278
```

### Phase 7: Update Exports

Update `olmix/model/__init__.py`:

```python
"""Model configuration module for olmix experiments."""

from olmix.model.aliases import ModelTrainConfig
from olmix.model.evaluators import (
    CodeTasks,
    DownstreamEvaluators,
    DownstreamEvaluatorsSmall,
)
from olmix.model.transformer import TransformerConfigBuilder

__all__ = [
    "CodeTasks",
    "DownstreamEvaluators",
    "DownstreamEvaluatorsSmall",
    "ModelTrainConfig",
    "TransformerConfigBuilder",
]
```

Create `olmix/utils/__init__.py`:

```python
from olmix.utils.cloud import expand_cloud_globs

__all__ = ["expand_cloud_globs"]
```

---

## Files to Modify

| File | Action |
|------|--------|
| `olmix/utils/cloud.py` | CREATE - glob expansion utility |
| `olmix/utils/__init__.py` | CREATE - exports |
| `olmix/model/aliases.py` | SIMPLIFY - keep only ModelTrainConfig |
| `olmix/model/transformer.py` | UPDATE - use olmo-core directly, inline MixtureBuilder |
| `olmix/model/__init__.py` | UPDATE - remove deleted exports |
| `olmix/data/dataset.py` | DELETE - MixtureBuilder removed |
| `olmix/data/__init__.py` | UPDATE - remove MixtureBuilder export |
| `tests/test_imports.py` | UPDATE - remove deleted import tests |
| `tests/conftest.py` | No changes needed |
| `configs/examples/*.yaml` | No changes needed (already correct) |

---

## Migration Guide

### For Code Using Removed Abstractions

```python
# Old way (DEPRECATED)
from olmix.model.aliases import SupportedModels, get_model_config, TOKENIZER
model = SupportedModels.olmo2_30m()
model = get_model_config("olmo2_30m")
tokenizer = TOKENIZER

# New way (RECOMMENDED)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.data import TokenizerConfig

tokenizer = TokenizerConfig.dolma2()
model = TransformerConfig.olmo2_30M(vocab_size=tokenizer.padded_vocab_size())
```

---

## Verification

```bash
# 1. Reinstall
cd /Users/kylel/ai2/olmix
uv pip install -e ".[dev]"

# 2. Test direct olmo-core usage
python -c "
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.data import TokenizerConfig

tokenizer = TokenizerConfig.dolma2()
print(f'Tokenizer vocab: {tokenizer.vocab_size}')

model = TransformerConfig.olmo2_30M(vocab_size=tokenizer.padded_vocab_size())
print(f'Model d_model: {model.d_model}')
print('Direct olmo-core usage works!')
"

# 3. Test glob utility
python -c "
from olmix.utils.cloud import expand_cloud_globs
# Test with non-glob path (should pass through)
paths = expand_cloud_globs(['s3://bucket/file.npy'])
print(f'Non-glob: {paths}')
"

# 4. Run tests
pytest tests/ -v

# 5. Test CLI
olmix --help
olmix launch validate --config configs/examples/test_olmo2_30m.yaml
```

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| `olmix/model/aliases.py` | 111 lines | ~30 lines |
| `olmix/data/dataset.py` | 87 lines | DELETED |
| Wrapper abstractions | 6 | 1 (`ModelTrainConfig`) |
| Direct olmo-core usage | Indirect | Direct |

**Lines removed:** ~170
**Abstractions removed:** 5 (`TOKENIZER`, `SupportedTokenizers`, `SupportedModels`, `get_model_config`, `MixtureBuilder`)
**Abstractions kept:** 2 (`ModelTrainConfig`, `TransformerConfigBuilder`)

---

## Implementation Status: COMPLETE

Implemented on: 2025-01-25

All phases completed successfully:
- Phase 1: Created `olmix/utils/cloud.py` with `expand_cloud_globs`
- Phase 2: Simplified `olmix/model/aliases.py` to only `ModelTrainConfig`
- Phase 3: Updated `olmix/model/transformer.py` with direct olmo-core usage
- Phase 4: Deleted `olmix/data/dataset.py`
- Phase 5: Config files unchanged (already correct)
- Phase 6: Updated tests
- Phase 7: Updated module exports

All 40 tests pass.

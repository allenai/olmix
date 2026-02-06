# Plan 002: Modernize olmix Model Configs to Match OLMo-core Style

## Overview

Update olmix model configurations to match the modern olmo-core style while keeping them defined within olmix. Pin olmo-core to a specific version and simplify the config code.

**Goals:**
1. Pin olmo-core to a specific branch/commit
2. Use `TokenizerConfig.dolma2()` tokenizer (vocab_size=100278)
3. Use `TransformerConfig` from olmo-core directly (remove custom `ModelConfig`)
4. Update naming to `olmo2_*` style
5. Simplify `olmix/model/` code

---

## Implementation Summary

### Phase 1: Pin olmo-core Version

Updated `pyproject.toml` to pin olmo-core to main branch:

```toml
"ai2-olmo-core @ git+https://github.com/allenai/OLMo-core.git@main",
```

### Phase 2: Simplify `olmix/model/aliases.py`

- Removed custom `ModelConfig` dataclass
- Changed `SupportedModels` from Enum to class with staticmethod factories
- Added `get_model_config()` for backward compatibility with legacy names
- Added `TOKENIZER = TokenizerConfig.dolma2()` as default

New model naming:
- `olmo2_1m` - Tiny test model (~1M params)
- `olmo2_30m` - Small model (~30M params)
- `olmo2_60m` - Medium-small model (~60M params)
- `olmo2_190m` - Medium model (~190M params)
- `olmo2_1b` - 1B model
- `olmo2_7b` - 7B model

### Phase 3: Update `olmix/model/transformer.py`

- Updated `TransformerConfigBuilder` to use `get_model_config()` instead of `SupportedModels[...].value`
- Removed field mapping in `build()` - now uses `TransformerConfig` directly
- Added constants `BATCH_DIVISOR = 32` and `SAVE_INTERVAL = 1000`

### Phase 4: Update References

- Updated `olmix/model/__init__.py` exports
- Updated `tests/test_imports.py` for new config structure
- Updated `tests/conftest.py` to use `olmo2_30m` and `dolma2` tokenizer

---

## Files Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | Pin olmo-core to @main |
| `olmix/model/aliases.py` | Remove ModelConfig, use TransformerConfig factories |
| `olmix/model/transformer.py` | Simplify to use TransformerConfig directly |
| `olmix/model/__init__.py` | Update exports |
| `tests/test_imports.py` | Update for new config structure |
| `tests/conftest.py` | Update proxy_model_id and tokenizer |

---

## Key Changes Summary

| Aspect | Before | After |
|--------|--------|-------|
| Config class | Custom `ModelConfig` | `TransformerConfig` (olmo-core) |
| Tokenizer | `gpt_neox` (50304) | `dolma2` (100278) |
| Naming | `olmo_30m` | `olmo2_30m` |
| Model lookup | `SupportedModels.olmo_30m.value` | `SupportedModels.olmo2_30m()` or `get_model_config("olmo2_30m")` |
| olmo-core dep | Unpinned | Pinned to @main |

---

## Backward Compatibility

Legacy model names still work via `get_model_config()`:

```python
# New style
config = SupportedModels.olmo2_30m()

# Legacy style (still works)
config = get_model_config("olmo_30m")  # Maps to olmo2_30m
```

---

## Verification

```bash
# 1. Reinstall
cd /Users/kylel/ai2/olmix
uv pip install -e ".[dev]"

# 2. Test imports
python -c "
from olmix.model.aliases import SupportedModels, get_model_config, TOKENIZER
print(f'Tokenizer vocab: {TOKENIZER.vocab_size}')
config = SupportedModels.olmo2_30m()
print(f'Model d_model: {config.d_model}')
config2 = get_model_config('olmo_30m')  # legacy
print(f'Legacy lookup works: {config2.d_model}')
"

# 3. Run tests
pytest tests/ -v

# 4. Test CLI
olmix --help
```

# Olmix

[![CI](https://github.com/allenai/olmix/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/olmix/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Olmix is a toolkit for optimizing pretraining data mixtures using regression-based proxy evaluation. It helps you find better data mixing ratios by learning from small-scale experiments (swarms) to predict performance and propose optimized mixtures for full-scale training runs.

## Overview

Training large language models requires carefully balancing data from different sources. Olmix implements the methodology from our data mixing research to:

1. **Fit regression models** from proxy experiments (swarms) that map data mixing ratios to downstream task performance
2. **Propose optimized mixtures** by finding mixing ratios that maximize predicted performance
3. **Analyze scaling laws** and data mixture effects across different tasks and domains

Key capabilities:
- Regression-based mixture optimization (LightGBM, log-linear models)
- Integration with W&B for experiment tracking
- Token-level constraints and repetition factor handling
- Pareto optimization for multi-objective balancing
- Support for both OLMo and custom training configurations

## Installation

### Prerequisites

Olmix uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. Install uv first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### From Source (Development)

Clone the repository and install in development mode:

```bash
git clone https://github.com/allenai/olmix.git
cd olmix
make install-dev
```

This will:
- Create a virtual environment using uv
- Install olmix in editable mode with all development dependencies
- Set up pre-commit hooks

Alternatively, install manually with uv:

```bash
# Install in development mode with all dependencies
uv pip install -e ".[dev]"

# Install for production use only
uv pip install -e .
```

### From PyPI

```bash
uv pip install olmix
```

## Quick Start

### Fitting a Regression Model

The main workflow involves fitting regression models from proxy experiments (swarms) tracked in Weights & Biases:

```bash
python -m olmix.fit.cli fit \
  --experiment-groups YOUR_WANDB_GROUP_ID \
  --config path/to/swarm_config.yaml \
  --workspace ai2-llm/your-workspace \
  --group-metrics all_metrics \
  --regression-type lightgbm
```

This will:
1. Pull experiment results from W&B
2. Calculate mixing ratios from each run
3. Fit regression models predicting task performance from mixing ratios
4. Propose an optimized mixture and save results to an output directory

### Key Options

- `--experiment-groups`: W&B group ID(s) for your swarm experiments
- `--config`: Path to your experiment configuration (defines data sources and constraints)
- `--regression-type`: Model type (`lightgbm`, `log_linear`, or `search`)
- `--group-metrics`: Which metrics to optimize (e.g., `all_metrics`, `mmlu_tasks`)
- `--opt-avg-metric`: Optimize the average across all metrics simultaneously
- `--constrain-objective`: Apply token availability constraints when proposing mixtures

See `python -m olmix.fit.cli fit --help` for all options.

## Core Concepts

### Swarms
A "swarm" is a collection of training runs where each run uses a different data mixture. By evaluating these runs on downstream tasks, we can learn how different mixing ratios affect performance.

### Regression Models
Olmix fits regression models (LightGBM or log-linear) that take data mixing ratios as input and predict task performance. This allows us to search for optimal mixtures without running expensive full-scale training.

### Configuration Files
Experiment configurations define:
- Data sources and their paths
- Token budgets and constraints
- Mixture sampling parameters (temperature, repetition factors)
- Source-level and topic-level mixing constraints

Example structure:
```yaml
name: my-swarm
sources:
  - name: common-crawl
    paths: ["s3://bucket/cc-data"]
    max_repetition_factor: 2.0
  - name: wikipedia
    paths: ["s3://bucket/wiki-data"]
    max_repetition_factor: 1.5
```

## Development

### Environment Management with uv

Olmix uses uv for all package management. Benefits:
- **Fast**: 10-100x faster than pip
- **Reliable**: Deterministic dependency resolution
- **Compatible**: Drop-in replacement for pip

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e ".[dev]"

# Add a new dependency
uv pip install new-package
uv pip freeze > requirements.txt
```

### Development Workflow

1. Clone and set up:
   ```bash
   git clone https://github.com/allenai/olmix.git
   cd olmix
   make install-dev
   ```

2. Make your changes and run checks:
   ```bash
   make format      # Format code
   make lint        # Check for issues
   make typecheck   # Type checking
   make test        # Run tests
   ```

3. Or run all checks at once:
   ```bash
   make run-checks
   ```

### Available Make Commands

```bash
make help         # Show all available commands
make install      # Install in production mode
make install-dev  # Install with development dependencies
make test         # Run all tests
make test-fast    # Run fast tests only
make test-cov     # Run tests with coverage
make format       # Format code with ruff
make lint         # Run linter (ruff)
make typecheck    # Run type checker (pyright)
make clean        # Clean build artifacts
make build        # Build distribution packages
make run-checks   # Run all quality checks
```

## Project Structure

```
olmix/
├── olmix/              # Main package directory
│   ├── __init__.py     # Package initialization
│   ├── version.py      # Version information
│   ├── aliases.py      # Type definitions and config schemas
│   ├── fit/            # Regression fitting and mixture optimization
│   │   ├── cli.py      # Command-line interface
│   │   ├── law.py      # Scaling law implementations
│   │   ├── utils.py    # Regression models and utilities
│   │   └── constants.py # Metric definitions
│   └── launch/         # Training launch utilities
│       ├── cli.py      # Launch commands
│       ├── launch_utils.py
│       └── synthesize_mixture.py
├── tests/              # Test files
├── scripts/            # Utility scripts
├── .github/            # GitHub Actions CI/CD
├── pyproject.toml      # Project configuration
├── Makefile            # Development tasks
└── README.md           # This file
```

## Testing

Run the test suite with uv:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run only fast tests (skip slow tests)
make test-fast
```

Tests are organized with pytest markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests requiring external resources
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.gpu` - Tests requiring GPU

## Advanced Usage

### Custom Optimization Objectives

Optimize for specific metric weights:
```bash
python -m olmix.fit.cli fit \
  --experiment-groups GROUP_ID \
  --config config.yaml \
  --obj-weights mmlu_heavy  # Use predefined weights
```

### Pareto Optimization

Ensure improvements over a reference model:
```bash
python -m olmix.fit.cli fit \
  --experiment-groups GROUP_ID \
  --config config.yaml \
  --dro-reference-model-id REFERENCE_RUN_ID \
  --tol 0.01  # Tolerance for Pareto constraint
```

### Constrained Search

Apply token budget constraints:
```bash
python -m olmix.fit.cli fit \
  --experiment-groups GROUP_ID \
  --config config.yaml \
  --constrain-objective \
  --final-cookbook-path path/to/final_config.yaml \
  --repetition-factor 1.5
```

### Log-Linear Regression with KL Regularization

```bash
python -m olmix.fit.cli fit \
  --experiment-groups GROUP_ID \
  --config config.yaml \
  --regression-type log_linear \
  --proposer-type exact \
  --kl-reg 0.1  # KL divergence penalty
```

## Migration from regmixer

This repository is a cleaned-up version of functionality from our internal `regmixer` repository. Key improvements:

- Modern Python packaging with `pyproject.toml`
- Faster dependency management with uv
- Cleaner separation of concerns
- Better documentation and type hints
- Removed training/launch code (focus on mixture optimization)

If you're migrating from regmixer, the core `fit` functionality remains similar, but with updated:
- Module paths (`olmix.fit` instead of `regmixer.eval`)
- Configuration schemas (see [aliases.py](olmix/aliases.py))
- Dependency management (uv instead of pip)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run quality checks (`make run-checks`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use Olmix in your research, please cite:

```bibtex
@software{olmix2024,
  title = {Olmix: Optimizing Pretraining Data Mixtures},
  author = {Lo, Kyle and others},
  year = {2024},
  url = {https://github.com/allenai/olmix}
}
```

## Acknowledgments

This project was developed at the [Allen Institute for AI](https://allenai.org/) as part of our work on the OLMo project and pretraining data optimization research.

## Support

For questions and support:
- Open an [issue](https://github.com/allenai/olmix/issues) on GitHub
- Check the [discussions](https://github.com/allenai/olmix/discussions) for Q&A

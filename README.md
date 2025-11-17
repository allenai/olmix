# Olmix

[![CI](https://github.com/allenai/olmix/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/olmix/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modern Python project for research and development, built with best practices and professional tooling.

## Features

- **Modern Python Packaging**: Uses `pyproject.toml` for all configuration
- **Fast Package Management**: Uses `uv` for lightning-fast dependency installation
- **Code Quality Tools**: Ruff for linting and formatting, Pyright for type checking
- **Testing Framework**: Comprehensive pytest setup with coverage reporting
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Pre-commit Hooks**: Automatic code formatting and quality checks

## Installation

### From Source (Development)

Clone the repository and install in development mode:

```bash
git clone https://github.com/allenai/olmix.git
cd olmix
make install-dev
```

### From PyPI

```bash
pip install olmix
```

## Quick Start

After installation, you can start using the package:

```python
import olmix

# Your code here
print(f"Olmix version: {olmix.__version__}")
```

## Development

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/allenai/olmix.git
   cd olmix
   ```

2. Install development dependencies:
   ```bash
   make install-dev
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Available Make Commands

```bash
make help         # Show all available commands
make test         # Run tests
make test-cov     # Run tests with coverage
make format       # Format code with ruff
make lint         # Run linter (ruff)
make typecheck    # Run type checker (pyright)
make clean        # Clean build artifacts
make run-checks   # Run all quality checks
```

## Project Structure

```
olmix/
├── olmix/              # Main package directory
│   ├── __init__.py     # Package initialization
│   └── version.py      # Version information
├── tests/              # Test files
│   ├── __init__.py
│   ├── conftest.py     # Pytest configuration
│   └── test_*.py       # Test modules
├── scripts/            # Utility scripts
├── .github/            # GitHub Actions workflows
│   └── workflows/
│       └── main.yml    # CI/CD pipeline
├── pyproject.toml      # Project configuration (all tools configured here)
├── Makefile            # Development tasks
├── README.md           # This file
├── CHANGELOG.md        # Version history
└── CONTRIBUTING.md     # Contribution guidelines
```

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run only fast tests
make test-fast
```

Tests are organized with pytest markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up your development environment
- Code style and standards
- Submitting pull requests
- Reporting issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Mayee Chen...
- ...

## Acknowledgments

This project was created at the [Allen Institute for AI](https://allenai.org/).

## Support

For questions and support:
- Open an [issue](https://github.com/allenai/olmix/issues) on GitHub
- Contact the maintainers at the emails listed above

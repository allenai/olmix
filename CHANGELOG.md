# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure based on modern Python best practices
- Comprehensive `pyproject.toml` configuration
- Development workflow with Makefile
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- Sphinx-based documentation setup
- Testing infrastructure with pytest
- Code formatting with black and isort
- Linting with ruff (replacing flake8)
- Type checking with mypy
- Professional README with badges and documentation

### Changed
- Migrated from setup.py-based configuration to pyproject.toml
- Updated minimum Python version to 3.11
- Replaced flake8 with ruff for faster linting
- Consolidated all tool configurations into pyproject.toml
- Line length standardized to 120 characters

### Removed
- Legacy configuration files (.flake8, mypy.ini, pytest.ini)
- Separate requirements.txt and dev-requirements.txt files

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Basic project structure

[Unreleased]: https://github.com/allenai/olmix/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/allenai/olmix/releases/tag/v0.1.0

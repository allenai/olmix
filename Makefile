.PHONY: help install install-dev test test-fast test-cov lint format format-check typecheck clean build docs serve-docs run-checks pre-commit all

# Default target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  install        Install the package in production mode"
	@echo "  install-dev    Install the package in development mode with all dependencies"
	@echo "  test           Run all tests with pytest"
	@echo "  test-fast      Run fast tests only (skip slow tests)"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run ruff linter and apply fixes"
	@echo "  format         Format code with ruff"
	@echo "  format-check   Check code formatting without applying changes"
	@echo "  typecheck      Run pyright type checker"
	@echo "  clean          Remove build artifacts and cache directories"
	@echo "  build          Build distribution packages"
	@echo "  run-checks     Run all code quality checks (lint, format-check, typecheck, test)"
	@echo "  pre-commit     Install and run pre-commit hooks"
	@echo "  all            Run clean, install-dev, run-checks, and build"

# Install package in production mode
install:
	uv pip install -e .

# Install package in development mode with all dependencies
install-dev:
	uv pip install -e ".[dev]"
	@echo "Installing pre-commit hooks..."
	-pre-commit install

# Run all tests
test:
	uvx pytest

# Run fast tests only (skip slow tests)
test-fast:
	uvx pytest -m "not slow"

# Run tests with coverage report
test-cov:
	uvx pytest --cov=olmix --cov-report=html --cov-report=term

# Run ruff linter with auto-fix
lint:
	uvx ruff check . --fix
	@echo "Linting complete!"

# Format code with ruff
format:
	uvx ruff format olmix tests scripts
	uvx ruff check --fix olmix tests scripts
	@echo "Formatting complete!"

# Check code formatting without applying changes
format-check:
	uvx ruff format --check olmix tests scripts
	uvx ruff check olmix tests scripts

# Run pyright type checker
typecheck:
	uvx pyright olmix

# Clean build artifacts and cache directories
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf docs/build/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".DS_Store" -delete

# Build distribution packages
build: clean
	python -m build
	@echo "Build complete! Check dist/ directory for packages."


# Run all code quality checks
run-checks:
	@echo "Running code quality checks..."
	@echo "1. Checking formatting..."
	@make format-check
	@echo ""
	@echo "2. Running linter..."
	@make lint
	@echo ""
	@echo "3. Running type checker..."
	@make typecheck
	@echo ""
	@echo "4. Running tests..."
	@make test
	@echo ""
	@echo "All checks passed!"

# Install and run pre-commit hooks
pre-commit:
	@if [ ! -f ".pre-commit-config.yaml" ]; then \
		echo ".pre-commit-config.yaml not found!"; \
		exit 1; \
	fi
	pre-commit install
	pre-commit run --all-files

# Run everything: clean, install, checks, and build
all: clean install-dev run-checks build
	@echo "All tasks completed successfully!"

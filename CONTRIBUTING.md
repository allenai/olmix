# Contributing to Olmix

Thank you for your interest in contributing to Olmix! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip
- make (optional but recommended)

### Setting Up Your Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/allenai/olmix.git
   cd olmix
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   make install-dev
   # Or without make:
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Standards

### Style Guide

We follow PEP 8 with the following modifications:
- Maximum line length: 120 characters
- Use double quotes for strings (enforced by black)

### Code Formatting

All code must be formatted using:
- **black** for Python code formatting
- **isort** for import sorting
- **ruff** for linting

Run formatting before committing:
```bash
make format
# Or manually:
black olmix tests scripts
isort olmix tests scripts
```

### Type Hints

- All functions must have type hints
- Use `from typing import` for type annotations
- Run mypy to check types:
  ```bash
  make typecheck
  # Or: mypy olmix
  ```

### Docstrings

Use Google-style docstrings for all public functions, classes, and modules:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed, explaining what the function
    does in more detail.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When validation fails
        TypeError: When types are incorrect
    """
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py` or `*_test.py`
- Use pytest fixtures for shared setup
- Mark tests appropriately:
  ```python
  @pytest.mark.unit
  def test_basic_functionality():
      """Test basic functionality."""
      pass

  @pytest.mark.integration
  def test_external_service():
      """Test integration with external service."""
      pass

  @pytest.mark.slow
  def test_large_dataset():
      """Test with large dataset (takes >1 second)."""
      pass
  ```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run only fast tests
make test-fast

# Run specific test file
pytest tests/test_specific.py

# Run tests matching pattern
pytest -k "test_pattern"
```

### Test Coverage

- Aim for at least 80% test coverage
- Focus on testing critical paths and edge cases
- View coverage reports:
  ```bash
  make test-cov
  open htmlcov/index.html  # View HTML coverage report
  ```

## Documentation

### Writing Documentation

- Update docstrings for all public APIs
- Add usage examples in docstrings
- Update README.md for significant changes
- Create tutorials for complex features

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make serve-docs
# Then open http://localhost:8000
```

## Submitting Changes

### Pre-submission Checklist

Before submitting a pull request, ensure:

- [ ] All tests pass: `make test`
- [ ] Code is formatted: `make format`
- [ ] No linting errors: `make lint`
- [ ] Type checks pass: `make typecheck`
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Commit messages follow conventions (see below)

### Commit Message Conventions

We use conventional commits format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

Examples:
```
feat(data): add CSV export functionality
fix(models): handle edge case in prediction
docs(readme): update installation instructions
test(utils): add tests for helper functions
```

### Pull Request Process

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a pull request on GitHub

5. Ensure all CI checks pass

6. Request review from maintainers

7. Address review feedback

8. Once approved, the PR will be merged

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Reference any related issues
- Provide a clear description of changes
- Include screenshots for UI changes

## Release Process

### For Maintainers

1. Update version in `olmix/version.py`
2. Update CHANGELOG.md with release notes
3. Create a git tag:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```
4. GitHub Actions will automatically:
   - Run tests
   - Build packages
   - Publish to PyPI (if configured)
   - Create GitHub release

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

## Questions and Support

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Contact maintainers: kylel@allenai.org

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## License

By contributing to Olmix, you agree that your contributions will be licensed under the Apache License 2.0.

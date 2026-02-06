"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test (fast, isolated)")
    config.addinivalue_line("markers", "integration: mark test as an integration test (may require external resources)")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files.

    Returns:
        pathlib.Path: Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def sample_data():
    """Provide sample data for testing.

    Returns:
        dict: Sample data dictionary
    """
    return {"name": "test", "value": 42, "items": ["a", "b", "c"], "nested": {"key1": "value1", "key2": "value2"}}


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables for each test.

    This fixture automatically runs before each test to ensure
    a clean environment.
    """
    # Add any environment variables that should be reset
    monkeypatch.setenv("TESTING", "true")
    yield
    # Cleanup happens automatically after the test


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing.

    Returns:
        dict: Mock configuration dictionary
    """
    return {
        "debug": True,
        "timeout": 30,
        "max_retries": 3,
        "api_url": "https://api.example.com",
    }


@pytest.fixture
def sample_experiment_config_dict():
    """Return a sample experiment config dictionary for olmix."""
    return {
        "name": "test-swarm",
        "description": "Test experiment",
        "budget": "ai2/oe-data",
        "workspace": "ai2/dolma2",
        "nodes": 1,
        "gpus": 1,
        "variants": 5,
        "max_tokens": 1000000,
        "sequence_length": 2048,
        "seed": 42,
        "cluster": "ai2/saturn-cirrascale",
        "tokenizer": "dolma2",
        "proxy_model_id": "olmo2_30m",
        "sources": [
            {"name": "wikipedia", "paths": ["s3://bucket/wiki/**/*.npy"]},
            {"name": "dclm", "paths": ["s3://bucket/dclm/**/*.npy"]},
        ],
    }


@pytest.fixture
def sample_experiment_config(sample_experiment_config_dict):
    """Return a sample ExperimentConfig for olmix."""
    from olmix.aliases import ExperimentConfig

    return ExperimentConfig(**sample_experiment_config_dict)


@pytest.fixture
def sample_experiment_config_file(sample_experiment_config_dict, tmp_path):
    """Create a sample config YAML file for olmix."""
    import yaml

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_experiment_config_dict, f)

    return config_file


@pytest.fixture
def sample_mixtures():
    """Return sample mixture weights."""
    return [
        {"wikipedia": (0.5, 1.0), "dclm": (0.5, 1.0)},
        {"wikipedia": (0.3, 1.5), "dclm": (0.7, 1.0)},
        {"wikipedia": (0.7, 1.0), "dclm": (0.3, 2.0)},
    ]


# ============================================================================
# Utility Functions for Tests
# ============================================================================


def assert_deep_equal(actual, expected, path=""):
    """Recursively assert deep equality of nested structures.

    Args:
        actual: The actual value
        expected: The expected value
        path: Current path in the structure (for error messages)
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"At {path}: expected dict, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), f"At {path}: keys mismatch"
        for key in expected:
            assert_deep_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, type(expected)), f"At {path}: type mismatch"
        assert len(actual) == len(expected), f"At {path}: length mismatch"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_deep_equal(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"At {path}: {actual} != {expected}"

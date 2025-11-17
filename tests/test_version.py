"""Tests for version information."""

import pytest

import olmix
from olmix.version import VERSION, VERSION_SHORT, __version__


@pytest.mark.unit
def test_version_format():
    """Test that version follows semantic versioning format."""
    parts = VERSION.split(".")
    assert len(parts) >= 3, "Version should have at least major.minor.patch"

    # Check that major and minor are numeric
    assert parts[0].isdigit(), "Major version should be numeric"
    assert parts[1].isdigit(), "Minor version should be numeric"

    # Patch might have suffixes like .dev or -alpha
    patch_numeric = parts[2].split("-")[0].split(".dev")[0]
    assert patch_numeric.isdigit(), "Patch version should start with a number"


@pytest.mark.unit
def test_version_short_format():
    """Test that short version contains only major.minor."""
    parts = VERSION_SHORT.split(".")
    assert len(parts) == 2, "Short version should be major.minor"
    assert parts[0].isdigit(), "Major version should be numeric"
    assert parts[1].isdigit(), "Minor version should be numeric"


@pytest.mark.unit
def test_version_consistency():
    """Test that VERSION and __version__ are the same."""
    assert VERSION == __version__, "VERSION and __version__ should match"


@pytest.mark.unit
def test_package_version():
    """Test that package exposes version correctly."""
    assert hasattr(olmix, "__version__"), "Package should expose __version__"
    assert olmix.__version__ == VERSION, "Package version should match VERSION"


@pytest.mark.unit
def test_version_short_in_version():
    """Test that VERSION starts with VERSION_SHORT."""
    assert VERSION.startswith(VERSION_SHORT), "VERSION should start with VERSION_SHORT"

"""
Pytest configuration for generated/batch tests.

These tests are auto-generated for coverage and are excluded from the default
test run. Run them explicitly with: pytest tests/generated/ -m generated

Or run all tests including generated: pytest tests/ --include-generated
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark all tests in this directory as 'generated'."""
    for item in items:
        # Add 'generated' marker to all tests in this directory
        if "generated" in str(item.fspath):
            item.add_marker(pytest.mark.generated)


def pytest_configure(config):
    """Register the 'generated' marker."""
    config.addinivalue_line(
        "markers",
        "generated: marks tests as auto-generated (deselect with '-m not generated')"
    )

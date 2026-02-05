"""Auto-apply the 'smoke' marker to all tests in this directory."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "/smoke/" in str(item.fspath) or "\\smoke\\" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)

"""
Test configuration file for pytest
"""
import pytest

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped in CI by default)")
    config.addinivalue_line("markers", "smoke: essential smoke tests that should always be run in CI")
    config.addinivalue_line("markers", "integration: integration tests requiring external dependencies")

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to apply appropriate markers.
    By default, all smoke tests will be considered 'not slow'.
    """
    for item in items:
        # If a test is marked as smoke, it should always run in CI
        if any(mark.name == "smoke" for mark in item.iter_markers()):
            continue  # Don't mark smoke tests as slow
            
        # Skip integration tests unless explicitly enabled
        if any(mark.name == "integration" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.slow) 
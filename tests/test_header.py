# tests/test_header.py
import pandas as pd
import pathlib
import sys
import pytest # Make sure pytest is in requirements if not already

# Dynamically import ALL_METRICS to avoid issues if test runs before PYTHONPATH is set
def get_all_metrics():
    try:
        flashcamp_root = pathlib.Path(__file__).parent.parent / "flashcamp"
        constants_path = flashcamp_root / "constants"
        sys.path.insert(0, str(flashcamp_root.parent)) # Add FLASH root
        import flashcamp.constants.metrics
        return flashcamp.constants.metrics.ALL_METRICS
    except (ImportError, AttributeError) as e:
        pytest.fail(f"Failed to import ALL_METRICS from flashcamp.constants.metrics: {e}")

def test_header_matches_contract():
    """Ensures the CSV header columns match the names in the constants list."""
    ALL_METRICS = get_all_metrics()
    expected_header = [m["name"] for m in ALL_METRICS]

    # Assuming test runs from FLASH root or has access relative to it
    csv_path = pathlib.Path("flashcamp/data/camp_plus_balanced_with_meta.csv")

    if not csv_path.is_file():
        pytest.fail(f"Data CSV file not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path, nrows=0)
        actual_header = df.columns.tolist()
    except Exception as e:
        pytest.fail(f"Failed to read CSV header from {csv_path}: {e}")

    # Use sets for easier comparison of differences
    expected_set = set(expected_header)
    actual_set = set(actual_header)

    missing_in_csv = expected_set - actual_set
    extra_in_csv = actual_set - expected_set

    error_msg = ""
    if missing_in_csv:
        error_msg += f"\nColumns in ALL_METRICS but MISSING in CSV header: {sorted(list(missing_in_csv))}"
    if extra_in_csv:
        error_msg += f"\nColumns in CSV header but NOT in ALL_METRICS: {sorted(list(extra_in_csv))}"

    # Compare sets instead of exact lists
    assert missing_in_csv == set(), f"Missing columns in CSV header!{error_msg}"
    assert extra_in_csv == set(), f"Extra columns in CSV header!{error_msg}"

print("✅ test_header.py created.") 
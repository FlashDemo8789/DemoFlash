#!/usr/bin/env python3
"""Exports the rich metrics list from constants to JSON for the frontend."""

import json, pathlib, sys

def export_metrics():
    try:
        # Assuming this script is run from the FLASH root
        flashcamp_root = pathlib.Path("flashcamp").resolve()
        constants_path = flashcamp_root / "constants"
        frontend_constants_path = flashcamp_root / "frontend" / "constants"
        metrics_py_path = constants_path / "metrics.py"
        metrics_json_out = frontend_constants_path / "metrics.json"

        # Add constants dir to sys.path to import module
        sys.path.insert(0, str(flashcamp_root.parent)) # Add FLASH root

        # Import the module dynamically
        import flashcamp.constants.metrics
        ALL_METRICS = flashcamp.constants.metrics.ALL_METRICS

        # Ensure frontend constants directory exists
        frontend_constants_path.mkdir(parents=True, exist_ok=True)

        # Write JSON output
        with open(metrics_json_out, "w") as f:
            json.dump(ALL_METRICS, f, indent=2)

        print(f"✅ metrics.json written to {metrics_json_out.relative_to(flashcamp_root.parent)} with {len(ALL_METRICS)} fields")

    except ImportError:
        print(f"Error: Could not import flashcamp.constants.metrics.", file=sys.stderr)
        print(f"Ensure {metrics_py_path} exists and PYTHONPATH includes the FLASH root directory.", file=sys.stderr)
        sys.exit(1)
    except AttributeError:
        print(f"Error: ALL_METRICS list not found in {metrics_py_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    export_metrics() 
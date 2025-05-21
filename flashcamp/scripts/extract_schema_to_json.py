#!/usr/bin/env python3
"""
Use once: extract current backend/schemas.py → contracts/metrics.json.
Promotes a legacy field list into JSON so product can tag required=True.

NOTE: This script assumes a pre-existing schemas.py file with a MetricsInput model.
It might need adjustments based on the exact structure of that legacy file.
It's primarily intended for a one-time conversion if you started without
the generator approach.
"""
import importlib.util, inspect, json, pathlib
import sys
from pydantic import Field
from typing import Optional, get_origin, get_args
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PY = ROOT / "backend" / "schemas.py"
DEST = ROOT / "contracts" / "metrics_extracted.json" # Save to a different name to avoid overwrite
DEST.parent.mkdir(exist_ok=True)

def python_type_to_json(py_type: type) -> str:
    """Convert Python type hint to JSON schema type string."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle Optional types
    if origin is Optional or origin is Union and type(None) in args:
        # Get the non-None type from Optional[T] or Union[T, None]
        actual_type = next(t for t in args if t is not type(None))
        return python_type_to_json(actual_type) # Recursive call for the base type

    # Handle basic types
    if py_type is int: return "integer"
    if py_type is float: return "number"
    if py_type is bool: return "boolean"
    if py_type is Decimal: return "currency"
    if py_type is str: return "string"

    # Handle list types
    if origin is list or origin is List:
        # Basic assumption: list of strings if no specific arg, might need refinement
        # arg_type = python_type_to_json(args[0]) if args else 'string'
        return "array" # JSON schema uses 'array'

    # Default or raise error for unhandled types
    print(f"Warning: Unhandled Python type: {py_type}. Defaulting to 'string'.")
    return "string"

def extract_schema():
    """Load MetricsInput from schemas.py and extract field info."""
    if not SCHEMA_PY.exists():
        print(f"Error: {SCHEMA_PY} not found. Cannot extract schema.")
        return None

    try:
        # Add backend directory to path to allow relative imports in schemas.py if any
        sys.path.insert(0, str(ROOT / "backend"))

        spec = importlib.util.spec_from_file_location("schemas", SCHEMA_PY)
        if spec is None or spec.loader is None:
             print(f"Error: Could not create module spec for {SCHEMA_PY}")
             return None
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Remove the path modification after import
        sys.path.pop(0)

        if not hasattr(mod, 'MetricsInput'):
             print(f"Error: MetricsInput class not found in {SCHEMA_PY}")
             return None

        return mod.MetricsInput.model_fields

    except Exception as e:
        print(f"Error loading or processing {SCHEMA_PY}: {e}")
        # Clean up sys.path if it was modified
        if str(ROOT / "backend") in sys.path:
             sys.path.pop(0)
        return None

def main():
    model_fields = extract_schema()
    if model_fields is None:
        exit(1)

    output_schema = {}
    for name, field_info in model_fields.items():
        # Determine if the field is required (not Optional)
        py_annotation = field_info.annotation
        is_required = not (get_origin(py_annotation) is Optional or \
                           (get_origin(py_annotation) is Union and type(None) in get_args(py_annotation)))

        # Get the base type if Optional
        base_type = py_annotation
        if not is_required:
            base_type = next(t for t in get_args(py_annotation) if t is not type(None))

        json_type = python_type_to_json(base_type)

        # Extract example from default if possible (excluding Ellipsis)
        example_val = None
        if field_info.default is not Ellipsis and field_info.default is not None:
            example_val = field_info.default
            # Convert Decimal example back to string for JSON consistency if needed
            if isinstance(example_val, Decimal):
                 example_val = str(example_val)

        output_schema[name] = {
            "type": json_type,
            "required": is_required,
            "description": field_info.description or "",
            # Only include example if it's not None
            **({"example": example_val} if example_val is not None else {})
        }

    try:
        DEST.write_text(json.dumps(output_schema, indent=2) + "\n")
        print(f"✅ Extracted {len(output_schema)} metrics schema → {DEST}")
        print("NOTE: Review the 'required' status and examples in the generated file.")
    except Exception as e:
        print(f"Error writing JSON to {DEST}: {e}")

if __name__ == "__main__":
    main() 
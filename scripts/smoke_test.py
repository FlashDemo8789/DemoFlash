import requests
import json
from pathlib import Path
import sys
import os

# Assuming the script is run from the project root (FLASH/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = PROJECT_ROOT / "flashcamp" / "contracts"
SPEC_PATH = CONTRACTS_DIR / "metrics.json"
API_URL = os.environ.get("FLASHDNA_API_URL", "http://localhost:8000/api/analyze")

def get_minimal_payload():
    """Generates a minimal valid payload based on the metrics.json contract."""
    try:
        with open(SPEC_PATH, 'r') as f:
            spec = json.load(f)
        
        payload = {}
        for key, definition in spec.items():
            if definition.get("required", False):
                # Use example value, or a sensible default if example is missing/invalid
                example = definition.get("example")
                if example is None:
                    # Basic type guessing for defaults
                    if definition.get("type") == "integer":
                        example = 0
                    elif definition.get("type") == "number":
                        example = 0.0
                    elif definition.get("type") == "string":
                        # Handle specific formats if needed, default to empty string
                        if definition.get("format") == "uuid":
                             # Generate a dummy UUID if needed, but often provided
                             example = "a1b2c3d4-e5f6-7890-1234-567890abcdef" 
                        elif definition.get("format") == "date-time":
                            example = "2024-01-01T00:00:00Z" # ISO 8601 format
                        else:
                            example = "string"
                    elif definition.get("type") == "boolean":
                        example = False
                    else:
                        example = "" # Fallback for other types
                
                payload[key] = example
                
        # Ensure startup_id is present and unique-ish for testing
        if 'startup_id' not in payload:
             payload['startup_id'] = "smoke-test-default-id"
        else:
            # Append a suffix to potentially make it unique per run
            import uuid
            payload['startup_id'] = f"{payload['startup_id']}-{uuid.uuid4().hex[:8]}"

        # Add any other mandatory fields if not covered by 'required' in spec
        # (Example: Add fields required by Pydantic model but not contract)
        # payload["some_other_required_field"] = default_value
        
        print(f"Generated Payload:\n{json.dumps(payload, indent=2)}")
        return payload

    except FileNotFoundError:
        print(f"Error: Specification file not found at {SPEC_PATH}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {SPEC_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating payload: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    payload = get_minimal_payload()
    
    print(f"\nSending request to {API_URL}...")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        print(f"\nResponse Status Code: {response.status_code}")
        
        try:
            response_data = response.json()
            print(f"Response Body:\n{json.dumps(response_data, indent=2)}")
            
            # Specific check for success probability
            if "success_probability" in response_data:
                print(f"\n---> Success Probability: {response_data['success_probability']:.3f}")
            else:
                print("\n---> Warning: 'success_probability' not found in response.")

        except json.JSONDecodeError:
            print("\nError: Could not decode JSON response.")
            print(f"Response Text:\n{response.text[:500]}...") # Print beginning of text

    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to the API at {API_URL}. Is the server running?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\nError: Request timed out connecting to {API_URL}.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\nError during request: {e}", file=sys.stderr)
        # Print response details if available
        if e.response is not None:
             print(f"Response Status Code: {e.response.status_code}")
             print(f"Response Body:\n{e.response.text[:500]}...")
        sys.exit(1) 
import json
import requests
import pprint
import sys
import uuid

# Load the metrics contract
try:
    spec = json.load(open("flashcamp/contracts/metrics_full.json"))
    print(f"Loaded contract with {len(spec)} metrics.")
except Exception as e:
    print(f"Failed to load contract: {e}")
    sys.exit(1)

# Create a minimal payload with just the required fields
payload = {k: v.get("example", 0) for k, v in spec.items() if v.get("required", False)}

# Add missing required fields
if "startup_id" not in payload:
    payload["startup_id"] = str(uuid.uuid4())
if "funding_stage" not in payload:
    payload["funding_stage"] = "Seed"
if "revenue_monthly_usd" not in payload:
    payload["revenue_monthly_usd"] = 27000

print(f"Created payload with {len(payload)} required fields.")

# Print the payload for verification
print("\nPayload:")
pprint.pprint(payload)

# Make the API request (uncomment when API is running)
print("\nMaking API request...")
try:
    response = requests.post("http://127.0.0.1:8000/api/analyze", json=payload)
    if response.status_code == 200:
        print(f"Success! Status code: {response.status_code}")
        result = response.json()
        
        # Print key parts of the response
        print("\nSuccess Probability:")
        print(result.get("success_probability"))
        
        print("\nPillar Scores:")
        pprint.pprint(result.get("pillar_scores"))
        
        # Check that we're not getting default values
        if result.get("success_probability") == 0.265:
            print("\n⚠️ WARNING: Got default success probability (0.265), model may not be working properly.")
        else:
            print("\n✅ Success probability is not the default value.")
    else:
        print(f"Error! Status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}") 
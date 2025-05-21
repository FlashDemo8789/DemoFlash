import json, pathlib, random

# Load existing contract to copy examples where available
try:
    existing_contract = json.load(open("flashcamp/contracts/metrics.json"))
except:
    existing_contract = {}

# Load feature names from model
cols = json.load(open("tmp/model_cols.json"))
contract = {}

# Create contract entries for each feature
for c in cols:
    # Initialize with defaults
    contract[c] = {
        "type": "number", 
        "required": False, 
        "description": f"Model feature: {c}"
    }
    
    # Copy example and better metadata from existing contract if available
    if c in existing_contract:
        for field in ["type", "description", "example"]:
            if field in existing_contract[c]:
                contract[c][field] = existing_contract[c][field]

# Flag core fields as required
core_fields = [
    "startup_id", "funding_stage", "team_size_full_time", "monthly_burn_usd",
    "revenue_monthly_usd", "cash_on_hand_usd", "runway_months",
    "market_growth_rate_percent", "founder_domain_experience_years"
]

for core in core_fields:
    if core in contract:
        contract[core]["required"] = True

# Ensure startup_id is set as string type
if "startup_id" in contract:
    contract["startup_id"]["type"] = "string"

# Write the new contract file
pathlib.Path("flashcamp/contracts/metrics_full.json").write_text(
    json.dumps(contract, indent=2)
)

print(f"Created metrics_full.json with {len(contract)} metrics.") 
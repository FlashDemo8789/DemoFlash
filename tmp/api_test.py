import json
import requests
import pprint
import uuid

print("Loading complete payload.json as example...")
try:
    example_payload = json.load(open("flashcamp/complete_payload.json"))
    print("Loaded example payload successfully")
except Exception as e:
    print(f"Could not load example payload: {e}")
    example_payload = {}

# Create a minimal payload with required fields
payload = {
    "startup_id": str(uuid.uuid4()),
    "funding_stage": "Seed",
    "team_size_full_time": 12,
    "monthly_burn_usd": 45000,
    "revenue_monthly_usd": 27000,
    "cash_on_hand_usd": 540000,
    "runway_months": 12,
    "market_growth_rate_percent": 18.5,
    "founder_domain_experience_years": 7
}

# Add a few more fields if we have examples
if example_payload:
    for field in ["sector", "subsector", "industry", "team_diversity_percent", "technical_team_size"]:
        if field in example_payload:
            payload[field] = example_payload[field]

print(f"Created payload with {len(payload)} fields")
print("\nPayload:")
pprint.pprint(payload)

print("\nMaking API request...")
try:
    url = "http://127.0.0.1:8003/api/analyze"
    print(f"Sending POST to {url}")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nSuccess! Status code: {response.status_code}")
        
        print("\nSuccess Probability:")
        print(result.get("success_probability"))
        
        print("\nPillar Scores:")
        pprint.pprint(result.get("pillar_scores"))
        
        if result.get("success_probability") == 0.265:
            print("\n⚠️ WARNING: Got default success probability (0.265)")
        else:
            print("\n✅ Success probability is NOT the default value")
            
        # Check for default pillar values
        pillar_scores = result.get("pillar_scores", {})
        default_scores = {
            "Capital": 0.50,
            "People": 0.85,
            "Market": 0.75,
            "Advantage": 0.60
        }
        
        matches_default = all(
            abs(pillar_scores.get(pillar, 0) - score) < 0.001
            for pillar, score in default_scores.items()
            if pillar in pillar_scores
        )
        
        if matches_default:
            print("⚠️ WARNING: Pillar scores match the default fallback values")
        else:
            print("✅ Pillar scores are NOT the default values")
    else:
        print(f"Error! Status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}") 
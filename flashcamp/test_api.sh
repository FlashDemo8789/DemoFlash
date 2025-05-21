#!/bin/bash

echo "Testing the FlashCAMP API..."

# Create a temporary JSON file with all field variations
cat > complete_payload.json << 'EOF'
{
  "startup_id": "test-123",
  "startup_name": "Test Company",
  "team_size": 10,
  "team_size_total": 10,
  "burn_rate_monthly": 50000,
  "monthly_burn_usd": 50000,
  "revenue_monthly": 30000,
  "monthly_revenue_usd": 30000,
  "cash_on_hand": 600000,
  "cash_on_hand_usd": 600000,
  "funding_stage": "Series A",
  "industry": "Technology",
  "country": "USA",
  "founding_year": 2023,
  "technical_team_size": 6
}
EOF

echo "Sending request to /api/analyze endpoint..."
curl -s -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d @complete_payload.json | jq '.'

echo "Done!" 
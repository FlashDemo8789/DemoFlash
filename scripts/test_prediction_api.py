#!/usr/bin/env python
"""
Test script for the hierarchical model API endpoints.
"""
import sys
import json
import requests
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parents[1]))

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust if your API runs on a different port

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    print("Testing prediction endpoint...")
    
    # Sample startup data
    startup_data = {
        "sector": "SaaS",
        "founders_count": 3,
        "team_size_total": 25,
        "domain_expertise_years_avg": 8.5,
        "previous_exits_count": 1,
        "board_advisor_experience_score": 7.2,
        "team_diversity_percent": 0.65,
        "gender_diversity_index": 0.7,
        "geography_diversity_index": 0.5,
        "key_person_dependency": 0.3,
        "cash_on_hand_usd": 2500000,
        "ltv_cac_ratio": 3.2,
        "burn_multiple": 1.8,
        "runway_months": 18,
        "gross_margin_percent": 68,
        "customer_concentration_percent": 15,
        "total_funding_usd": 5000000,
        "patents_count": 2,
        "has_network_effect": True,
        "has_data_moat": True,
        "regulatory_advantage_present": False,
        "tech_differentiation_score": 7.5,
        "switching_cost_score": 6.8,
        "brand_strength_score": 6.0,
        "product_retention_30d": 0.85,
        "product_retention_90d": 0.72,
        "nps_score": 45,
        "tam_size_usd": 15000000000,
        "sam_size_usd": 2500000000,
        "claimed_cagr_pct": 32,
        "market_growth_rate_percent": 28,
        "competition_intensity": 0.65,
        "top3_competitor_share_pct": 45,
        "industry_regulation_level": 0.4
    }
    
    try:
        # Make the API call
        response = requests.post(
            f"{API_BASE_URL}/api/prediction/predict",
            json=startup_data
        )
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            print("Prediction successful:")
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
            print("Pillar scores:")
            for pillar, score in result['pillar_scores'].items():
                print(f"  {pillar}: {score:.4f}")
            print(f"Final score: {result['final_score']:.4f}")
            print(f"Threshold: {result['threshold']:.4f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error calling prediction API: {e}")

def test_recommendations_endpoint():
    """Test the recommendations endpoint with sample data"""
    print("\nTesting recommendations endpoint...")
    
    # Same sample startup data
    startup_data = {
        "sector": "SaaS",
        "founders_count": 2,
        "team_size_total": 15,
        "domain_expertise_years_avg": 5.5,
        "previous_exits_count": 0,
        "board_advisor_experience_score": 4.2,
        "team_diversity_percent": 0.35,
        "gender_diversity_index": 0.4,
        "geography_diversity_index": 0.3,
        "key_person_dependency": 0.7,
        "cash_on_hand_usd": 1000000,
        "ltv_cac_ratio": 2.2,
        "burn_multiple": 2.8,
        "runway_months": 9,
        "gross_margin_percent": 58,
        "customer_concentration_percent": 35,
        "total_funding_usd": 2000000,
        "patents_count": 0,
        "has_network_effect": False,
        "has_data_moat": False,
        "regulatory_advantage_present": False,
        "tech_differentiation_score": 5.5,
        "switching_cost_score": 4.8,
        "brand_strength_score": 4.0,
        "product_retention_30d": 0.65,
        "product_retention_90d": 0.42,
        "nps_score": 25,
        "tam_size_usd": 5000000000,
        "sam_size_usd": 500000000,
        "claimed_cagr_pct": 18,
        "market_growth_rate_percent": 15,
        "competition_intensity": 0.85,
        "top3_competitor_share_pct": 75,
        "industry_regulation_level": 0.7
    }
    
    try:
        # Make the API call
        response = requests.post(
            f"{API_BASE_URL}/api/prediction/recommendations",
            json=startup_data
        )
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            print("Recommendations received:")
            for pillar, recs in result.items():
                print(f"\n{pillar.upper()} PILLAR:")
                for rec in recs:
                    print(f"  • {rec['recommendation']} (Impact: {rec['impact']})")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error calling recommendations API: {e}")

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\nTesting model-info endpoint...")
    
    try:
        # Make the API call
        response = requests.get(f"{API_BASE_URL}/api/prediction/model-info")
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            print("Model information:")
            print(f"Version: {result['model_version']}")
            print(f"Dataset size: {result['dataset_size']} samples")
            print(f"Success rate: {result['success_rate']:.2%}")
            print(f"Threshold: {result['threshold']:.4f}")
            
            print("\nMeta-model metrics:")
            for metric, value in result['meta_metrics'].items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error calling model-info API: {e}")

def test_visualization_endpoint():
    """Test the visualization endpoint with sample data"""
    print("\nTesting visualization endpoint...")
    
    # Sample startup data
    startup_data = {
        "sector": "SaaS",
        "founders_count": 3,
        "team_size_total": 25,
        "domain_expertise_years_avg": 8.5,
        "previous_exits_count": 1,
        "board_advisor_experience_score": 7.2,
        "team_diversity_percent": 0.65,
        "gender_diversity_index": 0.7,
        "geography_diversity_index": 0.5,
        "key_person_dependency": 0.3,
        "cash_on_hand_usd": 2500000,
        "ltv_cac_ratio": 3.2,
        "burn_multiple": 1.8,
        "runway_months": 18,
        "gross_margin_percent": 68,
        "customer_concentration_percent": 15,
        "total_funding_usd": 5000000,
        "patents_count": 2,
        "has_network_effect": True,
        "has_data_moat": True,
        "regulatory_advantage_present": False,
        "tech_differentiation_score": 7.5,
        "switching_cost_score": 6.8,
        "brand_strength_score": 6.0,
        "product_retention_30d": 0.85,
        "product_retention_90d": 0.72,
        "nps_score": 45,
        "tam_size_usd": 15000000000,
        "sam_size_usd": 2500000000,
        "claimed_cagr_pct": 32,
        "market_growth_rate_percent": 28,
        "competition_intensity": 0.65,
        "top3_competitor_share_pct": 45,
        "industry_regulation_level": 0.4
    }
    
    try:
        # Make the API call
        response = requests.post(
            f"{API_BASE_URL}/api/prediction/visualization",
            json=startup_data
        )
        
        # Check if successful
        if response.status_code == 200:
            # Save the image
            output_path = "reports/assets/api_visualization_test.png"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Visualization saved to {output_path}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error calling visualization API: {e}")

if __name__ == "__main__":
    # Run all tests
    test_prediction_endpoint()
    test_recommendations_endpoint()
    test_model_info_endpoint()
    test_visualization_endpoint()
    
    print("\nAll API tests completed.") 
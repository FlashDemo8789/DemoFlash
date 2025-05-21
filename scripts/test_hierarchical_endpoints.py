#!/usr/bin/env python
"""
Test script for the hierarchical model API endpoints.
"""
import sys
import os
import json
import requests
from typing import Dict, Any
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Default API URL - adjust as needed
API_URL = "http://localhost:8000"

# Sample startup data for testing
SAMPLE_STARTUP_DATA = {
    "startup_id": "test-123",
    "startup_name": "Test Startup",
    "sector": "SaaS",
    "founding_year": 2022,
    "team_size_total": 8,
    "funding_stage": "Seed",
    "runway_months": 12,
    "burn_multiple": 1.8,
    "cash_on_hand_usd": 500000,
    "ltv_cac_ratio": 3.2,
    "product_retention_30d": 0.7,
    "product_retention_90d": 0.5,
    "nps_score": 35,
    "tech_differentiation_score": 3.5,
    "brand_strength_score": 2.8,
    "switching_cost_score": 3.0,
    "network_effect_presence": 1,
    "market_growth_rate_percent": 25,
    "tam_size_usd": 5000000000,
    "competition_intensity": 3.5,
    "founder_domain_experience_years": 5,
    "prior_successful_exits_count": 1,
    "team_diversity_percent": 40,
    "domain_expertise_years_avg": 4.2,
    "key_person_dependency": 4,
}

def test_predict_endpoint():
    """Test the prediction endpoint"""
    print("\n--- Testing /api/prediction/predict endpoint ---")
    
    try:
        response = requests.post(
            f"{API_URL}/api/prediction/predict",
            json=SAMPLE_STARTUP_DATA
        )
        response.raise_for_status()
        result = response.json()
        
        # Validate response format
        expected_keys = [
            "pillar_scores", "final_score", "prediction", 
            "confidence", "threshold"
        ]
        
        for key in expected_keys:
            if key not in result:
                print(f"❌ Error: Missing expected key '{key}' in response")
                return False
        
        # Check pillar scores
        pillar_scores = result["pillar_scores"]
        for pillar in ["capital", "advantage", "market", "people"]:
            if pillar not in pillar_scores:
                print(f"❌ Error: Missing '{pillar}' in pillar_scores")
                return False
        
        print(f"✅ Success: Prediction response valid")
        print(f"   Final Score: {result['final_score']:.2f}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        
        for pillar, score in pillar_scores.items():
            print(f"   {pillar.capitalize()} Score: {score:.2f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing predict endpoint: {e}")
        return False

def test_recommendations_endpoint():
    """Test the recommendations endpoint"""
    print("\n--- Testing /api/prediction/recommendations endpoint ---")
    
    try:
        response = requests.post(
            f"{API_URL}/api/prediction/recommendations",
            json=SAMPLE_STARTUP_DATA
        )
        response.raise_for_status()
        result = response.json()
        
        # Validate response format
        for pillar in ["capital", "advantage", "market", "people"]:
            if pillar not in result:
                print(f"❌ Error: Missing '{pillar}' in recommendations")
                return False
            
            if not isinstance(result[pillar], list):
                print(f"❌ Error: '{pillar}' recommendations should be a list")
                return False
        
        # Count total recommendations
        total_recs = sum(len(recs) for recs in result.values())
        print(f"✅ Success: Received {total_recs} recommendations")
        
        # Print a few example recommendations
        for pillar, recs in result.items():
            if recs:
                print(f"   {pillar.capitalize()} recommendation example:")
                print(f"   - {recs[0]['recommendation']} (Impact: {recs[0]['impact']})")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing recommendations endpoint: {e}")
        return False

def test_visualization_endpoint():
    """Test the visualization endpoint"""
    print("\n--- Testing /api/prediction/visualization endpoint ---")
    
    try:
        response = requests.post(
            f"{API_URL}/api/prediction/visualization",
            json=SAMPLE_STARTUP_DATA
        )
        response.raise_for_status()
        
        # Check content type
        if response.headers.get("content-type") != "image/png":
            print(f"❌ Error: Expected content-type 'image/png', got '{response.headers.get('content-type')}'")
            return False
        
        # Try to open the image
        try:
            image = Image.open(BytesIO(response.content))
            print(f"✅ Success: Received visualization image ({image.width}x{image.height})")
            
            # Optionally save the image for inspection
            image_path = os.path.join(os.path.dirname(__file__), "visualization_test.png")
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"   Saved visualization to: {image_path}")
            
            return True
        except Exception as e:
            print(f"❌ Error parsing image: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Error testing visualization endpoint: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\n--- Testing /api/prediction/model-info endpoint ---")
    
    try:
        response = requests.get(f"{API_URL}/api/prediction/model-info")
        response.raise_for_status()
        result = response.json()
        
        # Validate response format
        expected_keys = [
            "model_version", "dataset_size", "success_rate", 
            "threshold", "pillar_metrics", "meta_metrics"
        ]
        
        for key in expected_keys:
            if key not in result:
                print(f"❌ Error: Missing expected key '{key}' in response")
                return False
        
        print(f"✅ Success: Model info response valid")
        print(f"   Model Version: {result['model_version']}")
        print(f"   Dataset Size: {result['dataset_size']}")
        print(f"   Success Rate: {result['success_rate']:.2f}")
        print(f"   Threshold: {result['threshold']:.2f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing model info endpoint: {e}")
        return False

def run_all_tests():
    """Run all endpoint tests"""
    print("=== Testing Hierarchical Model API Endpoints ===")
    print(f"API URL: {API_URL}")
    
    success_count = 0
    
    if test_predict_endpoint():
        success_count += 1
    
    if test_recommendations_endpoint():
        success_count += 1
    
    if test_visualization_endpoint():
        success_count += 1
    
    if test_model_info_endpoint():
        success_count += 1
    
    print(f"\n=== Test Results: {success_count}/4 endpoint tests passed ===")
    return success_count == 4

if __name__ == "__main__":
    # Allow specifying a different API URL
    if len(sys.argv) > 1:
        API_URL = sys.argv[1]
    
    success = run_all_tests()
    sys.exit(0 if success else 1) 
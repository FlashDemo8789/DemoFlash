"""
Test that feature preparation returns the correct number of features for each model.
"""
import pytest
import numpy as np
from flashcamp.backend.model_loader import load_model
from flashcamp.backend.features import prepare, feature_names_for

def test_features_match_model_expectations():
    """
    Test that prepare() returns exactly the number of features each model expects.
    """
    # Load the models
    models = {
        "success": load_model("success_xgb.joblib"),
        "people": load_model("pillar/people_lgbm.joblib"),
        "market": load_model("pillar/market_lgbm.joblib"),
        "capital": load_model("pillar/capital_lgbm.joblib"),
        "advantage": load_model("pillar/advantage_lgbm.joblib"),
    }
    
    # Sample input data with minimal required fields
    test_data = {
        "startup_id": "test-123",
        "funding_stage": "Seed",
        "team_size_full_time": 10,
        "monthly_burn_usd": 50000,
        "revenue_monthly_usd": 25000,
        "cash_on_hand_usd": 500000,
        "runway_months": 10,
        "market_growth_rate_percent": 15,
        "founder_domain_experience_years": 5
    }
    
    # Test each model
    for name, model in models.items():
        if model is None:
            pytest.skip(f"Model {name} could not be loaded")
            
        # Get expected feature count from model
        expected_features = feature_names_for(model)
        expected_count = len(expected_features)
        
        # Prepare features for this model
        X = prepare(test_data, model)
        
        # Check shape
        assert X.shape == (1, expected_count), f"Model {name} expected {expected_count} features, got {X.shape[1]}"
        print(f"✅ Model {name}: expected {expected_count} features, got {X.shape[1]}")

def test_default_prepare_uses_full_feature_list():
    """
    Test that prepare() without a model uses the full FEATURES list.
    """
    from flashcamp.backend.feature_map import FEATURES
    full_count = len(FEATURES)
    
    # Sample input data
    test_data = {"startup_id": "test-123"}
    
    # Prepare features without model
    X = prepare(test_data)
    
    # Check shape
    assert X.shape == (1, full_count), f"Expected {full_count} features, got {X.shape[1]}"
    print(f"✅ Default prepare: expected {full_count} features, got {X.shape[1]}") 
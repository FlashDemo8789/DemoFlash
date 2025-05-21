"""
Tests for the hierarchical model architecture.
Tests the hierarchical prediction process with both the actual models
and the fallback methods.
"""
import pytest
import numpy as np
import os
import json
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from flashcamp.backend.app.engines.ml import (
    predict_success_probability,
    _fallback_pillar_score,
    _fallback_prediction_with_pillars
)

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "gold"


@pytest.fixture
def sample_startup_data():
    """Create a sample startup data dictionary for testing"""
    return {
        # Capital metrics
        "cash_on_hand_usd": 2000000,
        "runway_months": 18,
        "burn_multiple": 1.5,
        "ltv_cac_ratio": 3.2,
        "gross_margin_percent": 65,
        "customer_concentration_percent": 20,
        "post_money_valuation_usd": 12000000,
        
        # Advantage metrics
        "patent_count": 2,
        "network_effects_present": True,
        "has_data_moat": True,
        "regulatory_advantage_present": False,
        "tech_differentiation_score": 4,
        "switching_cost_score": 3,
        "brand_strength_score": 2,
        "product_retention_30d": 0.85,
        "product_retention_90d": 0.65,
        "nps_score": 45,
        
        # Market metrics
        "tam_size_usd": 5000000000,
        "sam_size_usd": 500000000,
        "claimed_cagr_pct": 25,
        "market_growth_rate_percent": 18,
        "competition_intensity": 3,
        "top3_competitor_share_pct": 60,
        "industry_regulation_level": "medium",
        
        # People metrics
        "founders_count": 2,
        "team_size_total": 15,
        "founder_domain_experience_years": 8,
        "prior_successful_exits_count": 1,
        "board_advisor_experience_score": 3,
        "team_diversity_percent": 40,
        "gender_diversity_index": 0.45,
        "geography_diversity_index": 0.3,
        "key_person_dependency": False,
        
        # Additional metrics
        "sector": "SaaS",
        "product_stage": "GA",
        "investor_tier_primary": "Tier1"
    }


def test_fallback_pillar_scores(sample_startup_data):
    """Test that fallback pillar scores return reasonable values"""
    pillars = ["capital", "advantage", "market", "people"]
    
    for pillar in pillars:
        # Calculate the fallback score for this pillar
        score = _fallback_pillar_score(sample_startup_data, pillar)
        
        # Ensure the score is between 0 and 1
        assert 0 <= score <= 1
        
        # For a reasonably healthy startup like our sample, scores should be decent
        assert score >= 0.3, f"Fallback {pillar} score too low: {score}"


def test_fallback_prediction(sample_startup_data):
    """Test the fallback prediction method"""
    # Get the fallback prediction
    result = _fallback_prediction_with_pillars(sample_startup_data)
    
    # Check the structure of the result
    assert "success_probability" in result
    assert "pillar_scores" in result
    assert "error" in result
    
    # Check pillar scores
    pillar_scores = result["pillar_scores"]
    assert "capital" in pillar_scores
    assert "advantage" in pillar_scores
    assert "market" in pillar_scores
    assert "people" in pillar_scores
    
    # Check the success probability
    success_prob = result["success_probability"]
    assert 0 <= success_prob <= 1


def test_predict_success_responds_to_input_changes():
    """Test that the prediction changes when inputs change"""
    base_metrics = {
        "cash_on_hand_usd": 1000000,
        "runway_months": 12,
        "market_growth_rate_percent": 15,
        "team_size_total": 10
    }
    
    improved_metrics = base_metrics.copy()
    improved_metrics["cash_on_hand_usd"] = 5000000
    improved_metrics["runway_months"] = 24
    improved_metrics["market_growth_rate_percent"] = 30
    
    # Get predictions for both sets of metrics
    base_result = predict_success_probability(base_metrics)
    improved_result = predict_success_probability(improved_metrics)
    
    # Ensure both predictions return valid results
    assert "success_probability" in base_result
    assert "success_probability" in improved_result
    
    # The improved metrics should have a higher success probability
    assert improved_result["success_probability"] > base_result["success_probability"]
    
    # Also check that pillar scores changed
    assert improved_result["pillar_scores"]["capital"] > base_result["pillar_scores"]["capital"]
    assert improved_result["pillar_scores"]["market"] > base_result["pillar_scores"]["market"]


def test_full_prediction_output_structure(sample_startup_data):
    """Test the structure of the full prediction output"""
    # Get the full prediction
    result = predict_success_probability(sample_startup_data)
    
    # Check overall structure
    assert "success_probability" in result
    assert "pillar_scores" in result
    assert isinstance(result["success_probability"], float)
    assert isinstance(result["pillar_scores"], dict)
    
    # Check pillar scores
    for pillar in ["capital", "advantage", "market", "people"]:
        assert pillar in result["pillar_scores"]
        assert isinstance(result["pillar_scores"][pillar], float)
        assert 0 <= result["pillar_scores"][pillar] <= 1
    
    # Check that success probability is in valid range
    assert 0 <= result["success_probability"] <= 1


@pytest.mark.parametrize("change_factor", [0.5, 2.0])
def test_model_sensitivity(sample_startup_data, change_factor):
    """Test model sensitivity to key input changes"""
    # Make a copy of the original data
    modified_data = sample_startup_data.copy()
    
    # Apply the change factor to key metrics
    metrics_to_modify = [
        "cash_on_hand_usd",
        "runway_months",
        "market_growth_rate_percent",
        "tech_differentiation_score",
        "founder_domain_experience_years"
    ]
    
    for metric in metrics_to_modify:
        if metric in modified_data:
            # Don't modify boolean values
            if not isinstance(modified_data[metric], bool):
                modified_data[metric] = modified_data[metric] * change_factor
    
    # Get predictions
    original_result = predict_success_probability(sample_startup_data)
    modified_result = predict_success_probability(modified_data)
    
    # Verify that changes in input led to changes in output
    if change_factor > 1.0:
        assert modified_result["success_probability"] >= original_result["success_probability"]
    else:
        assert modified_result["success_probability"] <= original_result["success_probability"]


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 
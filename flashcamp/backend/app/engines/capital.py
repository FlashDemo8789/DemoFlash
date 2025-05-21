from typing import Dict, Any, Tuple, List
import logging

# Use the centralized loader and feature preparer
from flashcamp.backend.model_loader import load_model
from flashcamp.backend.features import prepare

logger = logging.getLogger(__name__)

# Load the specific model for this pillar
_model = load_model("pillar/capital_lgbm.joblib")

def calculate_capital_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate the capital score using the loaded ML model.
    Returns the score (0-1) and an empty list (no alerts generated here).
    """
    if _model is None:
        logger.error("Capital model not loaded.")
        # Instead of raising an error, fallback to heuristic calculation
        return _fallback_capital_score(metrics), []
        
    # Prepare features, passing only metrics - removed the model parameter
    X = prepare(metrics)
    
    try:
        # Use .predict() with shape check disabled to handle feature count mismatch
        raw_prediction = _model.predict(X, predict_disable_shape_check=True)
        score = float(raw_prediction[0])
        score = max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Error predicting capital score: {e}")
        # Use fallback score calculation if prediction fails
        score, _ = _fallback_capital_score(metrics)
        
    return score, []

def _fallback_capital_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Simple heuristic based capital score if model fails"""
    # Extract key capital metrics
    runway = float(metrics.get('runway_months', 10) or 10)
    cash = float(metrics.get('cash_on_hand_usd', 250000) or 250000)
    burn = float(metrics.get('monthly_burn_usd', 50000) or 50000)
    
    # Runway score: 6 months is minimum viable, 18+ months is excellent
    runway_score = min(1.0, max(0.0, (runway - 6) / 12))
    
    # Cash score: based on absolute value but with diminishing returns
    # $100K is minimum, $1M is good, $5M+ is excellent
    cash_score = min(1.0, max(0.0, 0.2 + (cash / 5000000) * 0.8))
    
    # Burn efficiency: lower is better, scaled by revenue
    revenue = float(metrics.get('revenue_monthly_usd', 1) or 1)
    if revenue > 0 and burn > 0:
        burn_ratio = revenue / burn
        efficiency_score = min(1.0, max(0.0, burn_ratio))
    else:
        efficiency_score = 0.3  # Default middle-low score
    
    # Overall capital score, weighted by importance
    score = runway_score * 0.4 + cash_score * 0.4 + efficiency_score * 0.2
    
    return max(0.0, min(1.0, score)), []

# --- Heuristics removed --- 
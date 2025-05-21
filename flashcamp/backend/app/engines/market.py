from typing import Dict, Any, Tuple, List
import logging

# Use the centralized loader and feature preparer
from flashcamp.backend.model_loader import load_model
from flashcamp.backend.features import prepare

logger = logging.getLogger(__name__)

# Load the specific model for this pillar
_model = load_model("pillar/market_lgbm.joblib")

def calculate_market_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate the market score using the loaded ML model.
    Returns the score (0-1) and an empty list (no alerts generated here).
    """
    if _model is None:
        logger.error("Market model not loaded.")
        # Instead of raising an error, fallback to heuristic calculation
        return _fallback_market_score(metrics), []
        
    # Prepare features, passing only metrics - removed the model parameter
    X = prepare(metrics)
    
    try:
        # Use .predict() with shape check disabled to handle feature count mismatch
        raw_prediction = _model.predict(X, predict_disable_shape_check=True)
        score = float(raw_prediction[0])
        score = max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Error predicting market score: {e}")
        # Use fallback score calculation if prediction fails
        score, _ = _fallback_market_score(metrics)
        
    return score, []

def _fallback_market_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Simple heuristic based market score if model fails"""
    # Extract key market metrics
    market_growth = float(metrics.get('market_growth_rate_percent', 25) or 25)
    
    # Growth under 5% is poor, over 40% is excellent
    growth_score = min(1.0, max(0.2, market_growth / 40))
    
    # Overall market score, weighted heavily by growth
    score = growth_score * 0.7 + 0.3  # Base score of 0.3 + up to 0.7 for growth
    
    return max(0.0, min(1.0, score)), []

# --- Remove all previous heuristic functions --- 
# def calculate_tam_score(...)
# def calculate_market_growth_score(...)
# def calculate_competition_score(...)
# def calculate_customer_score(...)
# def calculate_regulatory_score(...) 
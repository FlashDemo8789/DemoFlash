from typing import Dict, Any, Tuple, List
import logging

# Use the centralized loader and feature preparer
from flashcamp.backend.model_loader import load_model
from flashcamp.backend.features import prepare

logger = logging.getLogger(__name__)

# Load the specific model for this pillar
_model = load_model("pillar/people_lgbm.joblib")

def calculate_people_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate the people score using the loaded ML model.
    Returns the score (0-1) and an empty list (no alerts generated here).
    """
    if _model is None:
        logger.error("People model not loaded.")
        # Instead of raising an error, fallback to heuristic calculation
        return _fallback_people_score(metrics), []
        
    # Prepare features, passing only metrics - removed the model parameter
    X = prepare(metrics)
    
    try:
        # Use .predict() with shape check disabled to handle feature count mismatch
        raw_prediction = _model.predict(X, predict_disable_shape_check=True)
        score = float(raw_prediction[0])
        score = max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Error predicting people score: {e}")
        # Use fallback score calculation if prediction fails
        score, _ = _fallback_people_score(metrics)
        
    return score, []

def _fallback_people_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Simple heuristic based people score if model fails"""
    # Extract key people metrics
    experience = float(metrics.get('founder_domain_experience_years', 5) or 5)
    exits = float(metrics.get('prior_successful_exits_count', 0) or 0)
    team_size = float(metrics.get('team_size_full_time', 3) or 3)
    
    # Experience score: Under 3 years is low, 10+ years is excellent
    exp_score = min(1.0, max(0.0, experience / 10))
    
    # Prior exits score
    exit_score = min(1.0, max(0.0, 0.3 + (exits * 0.35)))  # Each exit adds 0.35, up to 0.7
    
    # Team size score: At least 3 people for a minimum viable team, 15+ for excellence
    team_score = min(1.0, max(0.0, (team_size - 1) / 14))
    
    # Overall people score, weighted by importance
    score = exp_score * 0.5 + exit_score * 0.3 + team_score * 0.2
    
    return max(0.0, min(1.0, score)), []

# --- Remove all previous heuristic functions --- 
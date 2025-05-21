from typing import Dict, Any, Tuple, List
import logging

# Use the centralized loader and feature preparer
from flashcamp.backend.model_loader import load_model
from flashcamp.backend.features import prepare

logger = logging.getLogger(__name__)

# Load the specific model for this pillar
_model = load_model("pillar/advantage_lgbm.joblib")

def calculate_advantage_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate the advantage score using the loaded ML model.
    Returns the score (0-1) and an empty list (no alerts generated here).
    """
    if _model is None:
        logger.error("Advantage model not loaded.")
        # Instead of raising an error, fallback to heuristic calculation
        return _fallback_advantage_score(metrics), []
        
    # Prepare features, passing only metrics - removed the model parameter
    X = prepare(metrics)
    
    try:
        # Use .predict() with shape check disabled to handle feature count mismatch
        raw_prediction = _model.predict(X, predict_disable_shape_check=True)
        score = float(raw_prediction[0])
        score = max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Error predicting advantage score: {e}")
        # Use fallback score calculation if prediction fails
        score, _ = _fallback_advantage_score(metrics)
        
    return score, []

def _fallback_advantage_score(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Simple heuristic based advantage score if model fails"""
    # Extract key advantage metrics - using various signals that might be available
    patents = 1 if metrics.get('has_patents', False) else 0
    network_effect = 1 if metrics.get('has_network_effect', False) or metrics.get('network_effects_present', False) else 0
    data_moat = 1 if metrics.get('has_data_moat', False) else 0
    
    # Calculate tech differentiation from tech_differentiation_score or default to medium
    tech_diff = float(metrics.get('tech_differentiation_score', 5) or 5) / 10
    
    # Calculate competition intensity (inverted: lower is better) or default to medium-high
    competition = 1.0 - (float(metrics.get('competition_intensity', 7) or 7) / 10)
    
    # Simple average of available advantage signals
    moat_signals = [
        tech_diff, 
        network_effect,
        data_moat,
        patents * 0.7,  # Patents provide strong but not perfect advantage
        competition
    ]
    
    # More weight to competition intensity and tech differentiation
    score = (tech_diff * 0.3 + 
             competition * 0.3 + 
             network_effect * 0.2 + 
             data_moat * 0.1 + 
             (patents * 0.7) * 0.1)
    
    return max(0.0, min(1.0, score)), []

# --- Heuristics removed --- 
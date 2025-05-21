"""
Analysis module for the backend.
Contains functions for calculating metrics, alerts, and predictive models.
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
import importlib.util
import sys
import os
import logging

# Define pillar names
PILLARS = ["Market", "Advantage", "People", "Capital"]

# Set up logging
logger = logging.getLogger(__name__)

def _import_engine_function(module_name, function_name):
    """Dynamically import function from engine modules"""
    try:
        # First try to import directly from the app.engines package
        try:
            # Try the direct import first
            module = importlib.import_module(f"flashcamp.backend.app.engines.{module_name}")
            return getattr(module, function_name)
        except (ImportError, ModuleNotFoundError):
            # Fall back to file-based import
            logger.debug(f"Direct import failed, trying file-based import for {module_name}.{function_name}")
            
            # Construct the path to the module
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            module_path = os.path.join(base_path, "flashcamp", "backend", "app", "engines", f"{module_name}.py")
            
            if not os.path.exists(module_path):
                # Try relative to current file
                module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "backend", "app", "engines", f"{module_name}.py")
            
            # Use importlib to dynamically load the module
            if os.path.exists(module_path):
                logger.debug(f"Found module at {module_path}")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Return the requested function
                return getattr(module, function_name)
            else:
                logger.warning(f"Engine module not found: {module_path}")
                return None
    except Exception as e:
        logger.error(f"Error importing engine function {function_name}: {e}")
        return None

def pillar_scores(metrics_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate scores for each pillar based on metrics data.
    
    Args:
        metrics_data: Dictionary with metric values
        
    Returns:
        Dictionary with calculated scores for each pillar
    """
    try:
        # Try to import the engine functions
        calculate_market_score_fn = _import_engine_function("market", "_fallback_market_score")
        calculate_advantage_score_fn = _import_engine_function("advantage", "_fallback_advantage_score")
        calculate_people_score_fn = _import_engine_function("people", "_fallback_people_score")
        calculate_capital_score_fn = _import_engine_function("capital", "_fallback_capital_score")
        
        # Check if we successfully imported the functions
        if all([calculate_market_score_fn, calculate_advantage_score_fn, 
                calculate_people_score_fn, calculate_capital_score_fn]):
            logger.info("Using engine implementations for calculating pillar scores")
            
            # Use the imported functions to calculate scores
            market_score, _ = calculate_market_score_fn(metrics_data)
            advantage_score, _ = calculate_advantage_score_fn(metrics_data)
            people_score, _ = calculate_people_score_fn(metrics_data)
            capital_score, _ = calculate_capital_score_fn(metrics_data)
            
            # Calculate overall score
            overall_score = (market_score + advantage_score + people_score + capital_score) / 4
            
            logger.info(f"Calculated pillar scores: Market={market_score:.2f}, Advantage={advantage_score:.2f}, " 
                       f"People={people_score:.2f}, Capital={capital_score:.2f}, Overall={overall_score:.2f}")
            
            return {
                "Market": market_score,
                "Advantage": advantage_score,
                "People": people_score,
                "Capital": capital_score,
                "Overall": overall_score
            }
    except Exception as e:
        logger.error(f"Error calculating pillar scores: {e}", exc_info=True)
    
    # Fall back to placeholder implementation if engines aren't available or errors occur
    logger.warning("Using placeholder pillar scores since engine calculation failed")
    return {
        "Market": 0.75,
        "Advantage": 0.60,
        "People": 0.85,
        "Capital": 0.50,
        "Overall": 0.68
    }

def imbalance_alert(pillar_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Check if there's a significant imbalance between pillar scores.
    
    Args:
        pillar_scores: Dictionary with calculated scores for each pillar
        
    Returns:
        List of alert details if imbalance detected, empty list otherwise
    """
    # Only consider the main pillars, not Overall
    pillars_to_check = [p for p in PILLARS if p in pillar_scores]
    
    if not pillars_to_check:
        return []
        
    min_pillar = min(pillars_to_check, key=lambda p: pillar_scores.get(p, 0))
    max_pillar = max(pillars_to_check, key=lambda p: pillar_scores.get(p, 0))
    
    alerts = []
    # Check for significant imbalance (>0.3 difference)
    if pillar_scores.get(max_pillar, 0) - pillar_scores.get(min_pillar, 0) > 0.3:
        alerts.append({
            "type": "pillar_imbalance",
            "message": f"Significant imbalance detected between {max_pillar} and {min_pillar}",
            "severity": "warning"
        })
    
    # Add alerts for low scores in specific pillars
    for pillar in pillars_to_check:
        score = pillar_scores.get(pillar, 0)
        if score < 0.4:
            alerts.append({
                "type": f"low_{pillar.lower()}_score",
                "message": f"Low {pillar} score ({score:.2f}) may be limiting overall success potential",
                "severity": "warning"
            })
    
    return alerts

def predict_success(metrics_data: Dict[str, Any]) -> float:
    """
    Predict success probability based on metrics data.
    
    Args:
        metrics_data: Dictionary with metric values
        
    Returns:
        Float representing success probability
    """
    try:
        # Try to import prediction function from ML engine
        predict_fn = _import_engine_function("ml", "predict_success_probability")
        
        if predict_fn:
            logger.info("Using ML model for success probability prediction")
            # Use the imported function to predict success probability
            probability = predict_fn(metrics_data)
            logger.info(f"Predicted success probability: {probability:.2f}")
            return probability
    except Exception as e:
        logger.error(f"Error predicting success probability: {e}", exc_info=True)
    
    # Fall back to placeholder implementation if engine isn't available or errors occur
    logger.warning("Using fallback success prediction method")
    
    # Import the fallback function from the ml engine
    try:
        fallback_fn = _import_engine_function("ml", "_fallback_prediction")
        if fallback_fn:
            fallback_result = fallback_fn(metrics_data)
            logger.info(f"Fallback prediction returned: {fallback_result:.2f}")
            return fallback_result
        else:
            logger.error("Could not import fallback prediction function!")
            # Fallback to a simple default if import fails
            return 0.5 
    except Exception as e:
        logger.error(f"Error during fallback prediction import/call: {e}", exc_info=True)
        return 0.5 # Final safety net 
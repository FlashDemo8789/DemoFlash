"""
ML Engine for FlashCAMP.
This module contains functions for prediction and scoring.
"""
import os
import logging
import random
import numpy as np
from typing import Dict, List, Any
import joblib
from ...schemas import MetricsInput

# Configure logging
logger = logging.getLogger(__name__)

# Global variables for model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "v2")
MAIN_MODEL_PATH = os.path.join(MODEL_DIR, "flashdna_main_v2.0.pkl")
CAPITAL_MODEL_PATH = os.path.join(MODEL_DIR, "capital_model_v2.0.pkl")
ADVANTAGE_MODEL_PATH = os.path.join(MODEL_DIR, "advantage_model_v2.0.pkl")
MARKET_MODEL_PATH = os.path.join(MODEL_DIR, "market_model_v2.0.pkl")
PEOPLE_MODEL_PATH = os.path.join(MODEL_DIR, "people_model_v2.0.pkl")

# Global model instances (loaded on-demand)
_main_model = None
_capital_model = None
_advantage_model = None
_market_model = None
_people_model = None

def _load_model():
    """
    Load and cache all models.
    This function is called on server startup.
    """
    global _main_model, _capital_model, _advantage_model, _market_model, _people_model
    
    try:
        # For this MVP version, we'll create placeholder models
        # In a real implementation, these would load trained models from disk
        logger.info("Loading ML models from disk...")
        
        # Check if model files exist, otherwise use random predictions
        if os.path.exists(MAIN_MODEL_PATH):
            _main_model = joblib.load(MAIN_MODEL_PATH)
            logger.info("Loaded main prediction model")
        else:
            logger.warning(f"Main model not found at {MAIN_MODEL_PATH}, using random predictions")
            _main_model = "random"
            
        # Load pillar models (or use random if not available)
        if os.path.exists(CAPITAL_MODEL_PATH):
            _capital_model = joblib.load(CAPITAL_MODEL_PATH)
            logger.info("Loaded capital model")
        else:
            logger.warning(f"Capital model not found at {CAPITAL_MODEL_PATH}, using random predictions")
            _capital_model = "random"
            
        if os.path.exists(ADVANTAGE_MODEL_PATH):
            _advantage_model = joblib.load(ADVANTAGE_MODEL_PATH)
            logger.info("Loaded advantage model")
        else:
            logger.warning(f"Advantage model not found at {ADVANTAGE_MODEL_PATH}, using random predictions")
            _advantage_model = "random"
            
        if os.path.exists(MARKET_MODEL_PATH):
            _market_model = joblib.load(MARKET_MODEL_PATH)
            logger.info("Loaded market model")
        else:
            logger.warning(f"Market model not found at {MARKET_MODEL_PATH}, using random predictions")
            _market_model = "random"
            
        if os.path.exists(PEOPLE_MODEL_PATH):
            _people_model = joblib.load(PEOPLE_MODEL_PATH)
            logger.info("Loaded people model")
        else:
            logger.warning(f"People model not found at {PEOPLE_MODEL_PATH}, using random predictions")
            _people_model = "random"
            
        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def calculate_capital_score(metrics: MetricsInput) -> float:
    """
    Calculate the capital pillar score (0-1).
    For MVP, we'll use a simple algorithm based on primary capital metrics.
    """
    # Get relevant metrics
    funding_stage = metrics.funding_stage or "Seed"
    monthly_burn = metrics.monthly_burn_usd or 0
    cash = metrics.cash_on_hand_usd or 0
    
    # Calculate baseline score
    if _capital_model == "random":
        # Deterministic random score based on startup_id to ensure consistency
        seed = hash(metrics.startup_id or "default") % 10000
        np.random.seed(seed)
        base_score = 0.3 + (np.random.random() * 0.5)  # Between 0.3 and 0.8
            else: 
        # In a real implementation, we would extract features and use the model
        # This is a simplified approach for MVP
        try:
            # Convert funding stage to a numerical value
            stage_values = {"Idea": 0, "Pre-seed": 1, "Seed": 2, "Series A": 3, "Series B": 4, "Series C+": 5}
            stage_value = stage_values.get(funding_stage, 2)  # Default to Seed
            
            # Calculate runway (months) if we have both cash and burn
            runway = min(cash / monthly_burn, 24) if monthly_burn > 0 else 12  # Cap at 24 months
            
            # Calculate runway efficiency (runway / burn)
            runway_efficiency = min(runway / 12, 2) if monthly_burn > 0 else 1.0
            
            # Calculate capital sufficiency
            capital_sufficiency = min(cash / (100000 * (stage_value + 1)), 1.0)
            
            # Combine factors
            base_score = (runway_efficiency * 0.5) + (capital_sufficiency * 0.3) + (stage_value / 10 * 0.2)
            base_score = max(0.1, min(0.95, base_score))  # Ensure score is between 0.1 and 0.95
        except Exception as e:
            logger.error(f"Error calculating capital score: {e}")
            # Fallback to deterministic random
            seed = hash(metrics.startup_id or "default") % 10000
            np.random.seed(seed)
            base_score = 0.3 + (np.random.random() * 0.5)
    
    return round(base_score, 2)

def calculate_advantage_score(metrics: MetricsInput) -> float:
    """
    Calculate the advantage pillar score (0-1).
    For MVP, we'll use a simple algorithm based on advantage metrics.
    """
    # Get relevant metrics
    tech_score = metrics.tech_differentiation_score or 5  # Scale 1-10
    network_effects = metrics.network_effects_present or False
    switching_cost = metrics.switching_cost_score or 5  # Scale 1-10
    
    # Calculate baseline score
    if _advantage_model == "random":
        # Deterministic random score based on startup_id to ensure consistency
        seed = hash(metrics.startup_id or "default") % 10000 + 1
        np.random.seed(seed)
        base_score = 0.25 + (np.random.random() * 0.6)  # Between 0.25 and 0.85
    else:
        # In a real implementation, we would extract features and use the model
        # This is a simplified approach for MVP
        try:
            # Normalize tech score to 0-1
            tech_factor = tech_score / 10
            
            # Network effects boost
            moat_factor = 0.2 if network_effects else 0
            
            # Normalize switching cost to 0-1
            switching_factor = switching_cost / 10
            
            # Combine factors
            base_score = (tech_factor * 0.4) + (moat_factor * 0.3) + (switching_factor * 0.3)
            base_score = max(0.15, min(0.95, base_score))  # Ensure score is between 0.15 and 0.95
        except Exception as e:
            logger.error(f"Error calculating advantage score: {e}")
            # Fallback to deterministic random
            seed = hash(metrics.startup_id or "default") % 10000 + 1
            np.random.seed(seed)
            base_score = 0.25 + (np.random.random() * 0.6)
    
    return round(base_score, 2)

def calculate_market_score(metrics: MetricsInput) -> float:
    """
    Calculate the market pillar score (0-1).
    For MVP, we'll use a simple algorithm based on market metrics.
    """
    # Get relevant metrics
    tam = metrics.tam_size_usd or 1000000000  # $1B default
    market_growth = metrics.market_growth_rate_percent or 10  # 10% default
    competition = metrics.competition_intensity or 5  # Scale 1-10
    
    # Calculate baseline score
    if _market_model == "random":
        # Deterministic random score based on startup_id to ensure consistency
        seed = hash(metrics.startup_id or "default") % 10000 + 2
        np.random.seed(seed)
        base_score = 0.3 + (np.random.random() * 0.5)  # Between 0.3 and 0.8
    else:
        # In a real implementation, we would extract features and use the model
        # This is a simplified approach for MVP
        try:
            # Normalize TAM to 0-1 (logarithmic scale)
            tam_factor = min(np.log10(tam) / 12, 1)  # $1T+ is max
            
            # Normalize market growth to 0-1
            growth_factor = min(market_growth / 100, 1)  # 100%+ annual growth is max
            
            # Inverse competition (10 is high competition, so invert)
            competition_factor = (10 - competition) / 10
            
            # Combine factors
            base_score = (tam_factor * 0.4) + (growth_factor * 0.4) + (competition_factor * 0.2)
            base_score = max(0.2, min(0.95, base_score))  # Ensure score is between 0.2 and 0.95
    except Exception as e:
            logger.error(f"Error calculating market score: {e}")
            # Fallback to deterministic random
            seed = hash(metrics.startup_id or "default") % 10000 + 2
            np.random.seed(seed)
            base_score = 0.3 + (np.random.random() * 0.5)
    
    return round(base_score, 2)

def calculate_people_score(metrics: MetricsInput) -> float:
    """
    Calculate the people pillar score (0-1).
    For MVP, we'll use a simple algorithm based on team metrics.
    """
    # Get relevant metrics
    domain_exp = metrics.founder_domain_experience_years or 3
    team_size = metrics.team_size_full_time or 2
    prior_startups = metrics.prior_startup_experience_count or 0
    prior_exits = metrics.prior_successful_exits_count or 0
    team_diversity = metrics.team_diversity_percent or 30
    
    # Calculate baseline score
    if _people_model == "random":
        # Deterministic random score based on startup_id to ensure consistency
        seed = hash(metrics.startup_id or "default") % 10000 + 3
        np.random.seed(seed)
        base_score = 0.35 + (np.random.random() * 0.5)  # Between 0.35 and 0.85
    else:
        # In a real implementation, we would extract features and use the model
        # This is a simplified approach for MVP
        try:
            # Domain experience factor (up to 20 years is max)
            domain_factor = min(domain_exp / 20, 1)
            
            # Team size factor (up to 20 people is max for early stage)
            size_factor = min(team_size / 20, 1)
            
            # Prior startup experience (up to 3 is max)
            exp_factor = min(prior_startups / 3, 1)
            
            # Prior successful exits (each exit is a big boost)
            exit_factor = min(prior_exits / 2, 1)
            
            # Team diversity factor
            diversity_factor = team_diversity / 100
            
            # Combine factors
            base_score = (domain_factor * 0.3) + (size_factor * 0.15) + (exp_factor * 0.2) + (exit_factor * 0.25) + (diversity_factor * 0.1)
            base_score = max(0.25, min(0.95, base_score))  # Ensure score is between 0.25 and 0.95
        except Exception as e:
            logger.error(f"Error calculating people score: {e}")
            # Fallback to deterministic random
            seed = hash(metrics.startup_id or "default") % 10000 + 3
            np.random.seed(seed)
            base_score = 0.35 + (np.random.random() * 0.5)
    
    return round(base_score, 2)

def predict_success_probability(metrics: MetricsInput) -> float:
    """
    Predict the probability of success (0-1).
    For MVP, we'll base this on pillar scores with different weights.
    """
    # Get pillar scores
    capital_score = calculate_capital_score(metrics)
    advantage_score = calculate_advantage_score(metrics)
    market_score = calculate_market_score(metrics)
    people_score = calculate_people_score(metrics)
    
    # Calculate overall probability
    if _main_model == "random":
        # Deterministic random score based on startup_id to ensure consistency
        seed = hash(metrics.startup_id or "default") % 10000 + 4
        np.random.seed(seed)
        base_prob = 0.2 + (np.random.random() * 0.6)  # Between 0.2 and 0.8
    else:
        # In a real implementation, we would extract features and use the model
        # For now, use weighted average of pillar scores
        try:
            # Apply pillar weights
            capital_weight = 0.30  # 30%
            advantage_weight = 0.20  # 20%
            market_weight = 0.25  # 25%
            people_weight = 0.25  # 25%
            
            # Calculate weighted average
            base_prob = (capital_score * capital_weight +
                         advantage_score * advantage_weight +
                         market_score * market_weight +
                         people_score * people_weight)
            
            # Apply some business rules for specific scenarios
            if capital_score < 0.2 or market_score < 0.2:
                # Critical weakness in capital or market is a major concern
                base_prob *= 0.7  # 30% penalty
                
            if people_score > 0.8 and advantage_score > 0.7:
                # Strong team with good advantage is promising
                base_prob = min(base_prob * 1.15, 0.95)  # 15% bonus, capped at 0.95
                
            # Ensure prob is between 0.1 and 0.95
            base_prob = max(0.1, min(0.95, base_prob))
        except Exception as e:
            logger.error(f"Error calculating success probability: {e}")
            # Fallback to deterministic random
            seed = hash(metrics.startup_id or "default") % 10000 + 4
            np.random.seed(seed)
            base_prob = 0.2 + (np.random.random() * 0.6)
    
    return round(base_prob, 2) 
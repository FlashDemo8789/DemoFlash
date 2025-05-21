"""
Backend app package initialization.
This file redirects imports to the main application to avoid duplication.
"""
import sys
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import engine functions for direct access
try:
    from .engines import (
        calculate_capital_score,
        calculate_advantage_score,
        calculate_market_score,
        calculate_people_score,
        predict_success_probability
    )
    
    logger.info("Successfully loaded engine functions")
except ImportError as e:
    logger.error(f"Error importing engine functions: {e}")
    raise
    
__all__ = [
    "calculate_capital_score",
    "calculate_advantage_score",
    "calculate_market_score",
    "calculate_people_score",
    "predict_success_probability"
] 
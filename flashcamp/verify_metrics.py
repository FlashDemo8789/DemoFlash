#!/usr/bin/env python
"""
Script to verify that model loading works and updates Prometheus metrics.
Run this with PYTHONPATH set to the project root.
"""
import os
import sys
import logging
import requests
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_model_loading():
    """Import the model directly and verify it loads"""
    try:
        # Set up environment variable for model path
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'success_xgb.joblib')
        os.environ['FLASHDNA_MODEL'] = model_path
        logger.info(f"FLASHDNA_MODEL set to: {model_path}")
        
        # Import and load the model
        from backend.app.engines.ml import _load_model
        model = _load_model()
        logger.info(f"Direct model load: {'SUCCESS' if model is not None else 'FAILED'}")
        return model is not None
    except Exception as e:
        logger.error(f"Error loading model directly: {e}")
        return False

def check_api(base_url="http://localhost:8000"):
    """Check if the API is running and responding"""
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            logger.info(f"API Check: SUCCESS - API is running")
            return True
        else:
            logger.warning(f"API Check: FAILED - Status code {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"API Check: FAILED - {e}")
        return False

def check_metrics(base_url="http://localhost:8000"):
    """Check if Prometheus metrics show model_loaded counter > 0"""
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code != 200:
            logger.warning(f"Metrics Check: FAILED - Status code {response.status_code}")
            return False
        
        # Check if our model_loaded metric is present
        metrics_text = response.text
        model_loaded_metrics = [
            line for line in metrics_text.split('\n')
            if 'flashcamp_model_loaded' in line and not line.startswith('#')
        ]
        
        logger.info(f"Found model_loaded metrics: {model_loaded_metrics}")
        
        # Check if model load count is at least 1
        if not model_loaded_metrics:
            logger.warning("Metrics Check: FAILED - No model_loaded metrics found")
            return False
            
        for line in model_loaded_metrics:
            try:
                metric_value = float(line.split()[1])
                if metric_value >= 1:
                    logger.info(f"Metrics Check: SUCCESS - Model load count is {metric_value}")
                    return True
            except (IndexError, ValueError):
                pass
                
        logger.warning("Metrics Check: FAILED - Model load count is 0")
        return False
    except requests.RequestException as e:
        logger.error(f"Metrics Check: FAILED - {e}")
        return False

def main():
    """Main function to verify model loading and metrics"""
    # 1. Verify direct model loading
    direct_load = verify_model_loading()
    
    # 2. Check if API is up
    api_up = check_api()
    
    if api_up:
        # 3. Check metrics
        metrics_valid = check_metrics()
        
        if direct_load and metrics_valid:
            logger.info("ALL CHECKS PASSED: Model loads correctly and metrics are tracked")
            return 0
        else:
            logger.warning(f"SOME CHECKS FAILED: Direct load: {direct_load}, Metrics valid: {metrics_valid}")
            return 1
    else:
        logger.warning("API is not running, can't check metrics")
        if direct_load:
            logger.info("Direct model loading works, but API is not available")
            return 0
        else:
            return 1

if __name__ == "__main__":
    # Ensure PYTHONPATH includes the project root
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    sys.exit(main()) 
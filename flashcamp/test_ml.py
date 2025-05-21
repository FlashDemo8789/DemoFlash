import os
import sys
import logging

# Set up environment variable
os.environ['FLASHDNA_MODEL'] = os.path.join(os.path.dirname(__file__), 'models', 'success_xgb.joblib')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modify sys.path to include the current directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the ml module directly
from flashcamp.backend.app.engines.ml import _load_model, MODEL_PATH

def test_model_loading():
    """Test if the model loads correctly"""
    model = _load_model()
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Model path used: {MODEL_PATH}")
    return model is not None

if __name__ == "__main__":
    # Print the environment variables
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"FLASHDNA_MODEL: {os.environ.get('FLASHDNA_MODEL', 'Not set')}")
    
    # Test model loading
    success = test_model_loading()
    print(f"Model loaded successfully: {success}")
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 
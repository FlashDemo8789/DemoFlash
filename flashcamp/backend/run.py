"""
Script to run the FastAPI application with proper configuration
Handles import paths and ensures the correct application is launched
"""
import os
import sys
import uvicorn
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to enable proper imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def initialize_app():
    """Perform initialization tasks before starting the server"""
    try:
        # Import initialization functions here to avoid circular imports
        from flashcamp.backend.database import init_db
        
        # Initialize database
        logger.info("Initializing application...")
        init_db()
        logger.info("Database initialization complete")
        
        # Check for required directories and create them if missing
        required_dirs = [
            os.path.join(parent_dir, "flashcamp", "models"),
            os.path.join(parent_dir, "flashcamp", "reports", "pdf"),
            os.path.join(parent_dir, "flashcamp", "logs")
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
                
        # Check if model files exist
        model_dir = os.path.join(parent_dir, "flashcamp", "models")
        model_files = os.listdir(model_dir)
        if model_files:
            logger.info(f"Found model files: {', '.join(model_files)}")
        else:
            logger.warning("No model files found in models directory")
            
        logger.info("Initialization complete")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting FlashDNA API server")
    
    # Run initialization
    initialize_app()
    
    # Use the app from app.py, not app/main.py
    # The updated analysis.py will use the engine implementations
    uvicorn.run(
        "flashcamp.backend.app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    ) 
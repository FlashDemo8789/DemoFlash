"""
This module has been deprecated in favor of the main app.py implementation.
It remains here for backward compatibility but redirects to the primary app instance.
"""
import logging
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Issue a deprecation warning
warnings.warn(
    "The app/main.py implementation is deprecated. Please use the main app.py instance instead.",
    DeprecationWarning,
    stacklevel=2
)
logger.warning("Using deprecated app/main.py - this file will be removed in a future update")

# Import the main application instance
from ..app import app

# Re-export the app instance
__all__ = ["app"]

# When this file is run directly, forward to the main app.py
if __name__ == "__main__":
    import uvicorn
    from ...config import settings
    
    logger.info("Starting deprecated app/main.py - redirecting to main app.py")
    
    # Use the main app instance
    uvicorn.run(
        "flashcamp.backend.app:app", 
        host=settings.HOST, 
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development"
    ) 
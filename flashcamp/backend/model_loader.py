import joblib
import pathlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_model_cache = {}
BASE_MODEL_DIR = pathlib.Path(__file__).parents[1] / "models"

def load_model(relative_path: str) -> Any:
    """
    Loads a model from the specified path relative to the models directory,
    using a cache to avoid reloading.
    Handles both .joblib files.
    """
    if relative_path in _model_cache:
        return _model_cache[relative_path]

    absolute_path = BASE_MODEL_DIR / relative_path
    model = None
    try:
        logger.info(f"Loading model from {absolute_path}")
        if not absolute_path.exists():
             raise FileNotFoundError(f"Model file not found: {absolute_path}")

        if absolute_path.suffix == ".joblib":
            model = joblib.load(absolute_path)
            logger.info(f"Loaded model {absolute_path} successfully.")
        else:
            logger.error(f"Unsupported model file type: {absolute_path.suffix} for {absolute_path}")
            raise ValueError(f"Unsupported model file type: {absolute_path.suffix}")

        _model_cache[relative_path] = model
        return model

    except FileNotFoundError:
         logger.error(f"Model file not found at {absolute_path}. Returning None.")
         # Decide: raise error or return None? Returning None allows fallback.
         _model_cache[relative_path] = None # Cache the failure
         return None
    except Exception as e:
        logger.exception(f"Error loading model {absolute_path}: {e}")
        _model_cache[relative_path] = None # Cache the failure
        return None

# Example pre-loading (optional, could be done in main app startup)
# load_model("success_xgb.joblib")
# load_model("pillar/market_lgbm.joblib")
# ... etc 
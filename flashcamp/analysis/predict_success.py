"""
Predictive Success Stack  (Pr-1)
────────────────────────────────
Loads the trained XGBoost model `success_xgb.joblib` (already in
flashcamp/models/) and outputs a probability 0-1.

If the file is missing (e.g. fresh clone without models), it degrades
gracefully by averaging the four pillar scores.
"""

from pathlib import Path
from joblib import load
import numpy as np
# Import pillar_scores function directly to ensure it's available
from flashcamp.analysis import pillar_scores
import json


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "success_xgb.joblib"
_model = None

# Attempt to load the model during module initialization
if MODEL_PATH.is_file(): # Check if file exists first
    try:
        _model = load(MODEL_PATH)
        print(f"✅ Successfully loaded success model: {MODEL_PATH.name}")
    except ImportError as e:
        # Specifically catch import errors (e.g., missing xgboost)
        print(f"[warn] Could not load {MODEL_PATH.name} due to missing dependency: {e}. Falling back to pillar average.")
    except Exception as e:  # Catch other potential errors (corrupt pickle, etc.)
        print(f"[warn] Could not load {MODEL_PATH.name}: {e}. Falling back to pillar average.")
else:
    print(f"[info] Model file not found: {MODEL_PATH}. Falling back to pillar average for success probability.")

# Assign loaded data to METRICS_ALL
METRICS_PATH = Path(__file__).resolve().parent.parent / "frontend" / "src" / "constants" / "metrics.json"
if not METRICS_PATH.is_file():
    print(f"Error: Cannot find metrics.json at {METRICS_PATH} or {Path(__file__).resolve().parent.parent / 'frontend' / 'constants' / 'metrics.json'}")
    import sys; sys.exit(1) # Add import sys

# Assign loaded data to METRICS_ALL
METRICS_ALL = json.loads(METRICS_PATH.read_text())

# Filter metrics to include only numeric and boolean types for XGBoost
METRICS_FOR_XGB = [m["name"] for m in METRICS_ALL if m["type"] in ["number", "checkbox"]]

_FEATURES_OLD = ["capital", "advantage", "market", "people"]  # Old pillar names

def probability(payload: dict) -> float:
    """
    Returns calibrated P(success) in [0,1].
    Falls back to mean pillar score / 100.
    """

    if _model is None:
        # Fallback logic: average numeric pillar scores
        # Calculate pillars only if needed for fallback
        pillars = pillar_scores(payload)
        def _to_num(val):
            if isinstance(val, (int, float)): return val
            cmap = {"Green": 80, "Light-green": 70, "Amber": 50, "Red": 30, "Deep-red": 20}
            return cmap.get(val, 50)
            
        numeric_scores = [_to_num(v) for v in pillars.values()]
        # Average the numeric representations (0-100 scale from _to_num)
        prob = sum(numeric_scores) / (len(numeric_scores) * 100) if numeric_scores else 0.25 # Default 0.25 if no numeric scores
        return round(prob, 3)

    # --- Assemble feature vector expected by the TRAINED model ---
    try:
        print(f"[debug] Attempting prediction with {len(METRICS_FOR_XGB)} expected features.") # DEBUG PRINT
        # 1. Extract the 79 features used during training directly from the payload
        # Ensure correct types (float for number, bool for checkbox)
        feature_values = []
        for metric_name in METRICS_FOR_XGB:
            value = payload.get(metric_name)
            metric_info = next((m for m in METRICS_ALL if m["name"] == metric_name), None)
            metric_type = metric_info.get("type") if metric_info else None

            if metric_type == "number":
                try:
                    # Convert to float, handle None/empty string as NaN
                    num_val = float(value) if value is not None and value != "" else np.nan
                    feature_values.append(num_val)
                except (ValueError, TypeError):
                    feature_values.append(np.nan) # Assign NaN if conversion fails
            elif metric_type == "checkbox":
                # Convert to boolean (handle strings 'true', 'false', 1, 0 etc.)
                bool_val = str(value).lower() in ['true', '1', 'yes']
                feature_values.append(bool_val)
            else:
                # Should not happen if METRICS_FOR_XGB is correct, but handle defensively
                feature_values.append(np.nan)

        # 2. Create NumPy array and handle potential NaNs (e.g., fill with median)
        x = np.array([feature_values])
        print(f"[debug] Constructed feature vector with shape: {x.shape}") # DEBUG PRINT
        # Check for NaNs introduced by missing values or conversion errors
        if np.isnan(x).any():
             # Find median *from the training data* ideally, but for now use 0 as simple imputation
             # A more robust approach would load training set medians
             print("[warn] NaNs detected in feature vector during prediction. Imputing with 0.")
             x = np.nan_to_num(x, nan=0.0) 

        # 3. Predict probability using the loaded model
        prob = float(_model.predict_proba(x)[:, 1]) # Assumes binary classification model
        return round(prob, 3)
    except Exception as e:
        print(f"[error] Failed during model prediction: {e}. Falling back to pillar average.")
        # Fallback logic again in case prediction fails
        pillars = pillar_scores(payload) # Calculate pillars for fallback
        def _to_num(val):
            if isinstance(val, (int, float)): return val
            cmap = {"Green": 80, "Light-green": 70, "Amber": 50, "Red": 30, "Deep-red": 20}
            return cmap.get(val, 50)
        numeric_scores = [_to_num(v) for v in pillars.values()]
        prob = sum(numeric_scores) / (len(numeric_scores) * 100) if numeric_scores else 0.25
        return round(prob, 3) 
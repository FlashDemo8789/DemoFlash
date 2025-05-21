"""
Build a numeric vector that matches flashcamp.backend.feature_map.FEATURES.

Handles:
 • scalar numeric/boolean/string ↦ float
 • *_len synthetic features for array-type metrics
 • on-the-fly one-hot columns of pattern family_option

The function is model-agnostic; pillar engines may slice the first N cols
if their LightGBM booster expects fewer than the master 99.
"""
from __future__ import annotations
from typing import Any, List
import numpy as np
import re, json, logging
from flashcamp.backend.feature_map import FEATURES

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_CURRENCY = re.compile(r"_usd$", re.I)

# Validation function
def _validate_input(raw: dict[str, Any]) -> None:
    """Log warnings for missing or unexpected fields in raw input."""
    # Check for missing backend features
    for feature in FEATURES:
        base_feature = feature[:-4] if feature.endswith('_len') else feature
        if '_' in feature and not feature.endswith('_len'):
            base_feature = feature.split('_', 1)[0]
        
        if base_feature not in raw and not any(
            f.startswith(base_feature + '_') for f in FEATURES if '_' in f
        ):
            logger.warning(f"Missing input for backend feature: {feature} (expected frontend field: {base_feature})")
    
    # Check for unexpected frontend fields
    for frontend_field in raw.keys():
        if not any(
            f == frontend_field or 
            f.startswith(frontend_field + '_') or 
            f == frontend_field + '_len'
            for f in FEATURES
        ):
            logger.warning(f"Unexpected frontend field: {frontend_field} (no matching backend feature)")

def _coerce_scalar(v: Any) -> float:
    if v in ("", None):                   return 0.0
    if isinstance(v, bool):               return 1.0 if v else 0.0
    try:    return float(str(v).replace("$", "").replace(",", ""))
    except Exception: return 0.0

def _array_len(v: Any) -> float:
    if v in ("", None):                   return 0.0
    if isinstance(v, (list, tuple, set)): return float(len(v))
    # tolerate JSON-ish strings: "['a','b']" or "a,b"
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return float(len(parsed))
        except Exception:
            return float(len([x for x in v.split(",") if x.strip()]))
        return 0.0

def _one_hot(fam: str, cat: str, raw: dict[str, Any]) -> float:
    return float(str(raw.get(fam, "")).lower() == cat.lower())

def prepare(raw: dict[str, Any]) -> np.ndarray:
    """
    Parameters
    ----------
    raw : dict
        already validated MetricsInput dict

    Returns
    -------
    np.ndarray shape (1, len(FEATURES))  — strictly ordered
    """
    # Validate input before processing
    _validate_input(raw)
    
    vec: List[float] = []

    for col in FEATURES:
        if col.endswith("_len"):
            fam = col[:-4]                # strip "_len"
            vec.append(_array_len(raw.get(fam)))
            continue

        if "_" in col and col.split("_", 1)[0] in raw:
            fam, cat = col.split("_", 1)
            vec.append(_one_hot(fam, cat, raw))
            continue

        vec.append(_coerce_scalar(raw.get(col)))
    
    # Log the feature vector shape for debugging
    feature_array = np.asarray([vec], dtype=np.float32)
    logger.info(f"Prepared feature array with shape: {feature_array.shape}")
    
    return feature_array 
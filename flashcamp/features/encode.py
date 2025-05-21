"""
Reusable encoders & helpers used across training and inference.
All are pure-Python / NumPy to avoid fitting state during inference.
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json

# ----- Ordinal mappings -----------------------------------------------------
SWITCHING_COST_IDX = {"Low": 0, "Medium": 0.5, "High": 1.0}
# Updated to match numeric values in new dataset (0-3)
REG_LEVEL_IDX      = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0} 
DEGREE_IDX         = {"None": 0, "Bachelor": 0.3, "Master": 0.6, "PhD": 1.0}
# New mapping for investor tiers
INVESTOR_TIER_IDX  = {"Unknown": 0.0, "Angel": 0.33, "Tier2": 0.66, "Tier1": 1.0}
# New mapping for product stages
PRODUCT_STAGE_IDX  = {"Concept": 0.0, "Beta": 0.5, "GA": 1.0}

def safe_div(num: float, denom: float) -> float:
    """Safe division to handle zero denominators"""
    return 0.0 if denom == 0 else num / denom

# ----- One-hot helpers -------------------------------------------------------
_CACHED_OHE = {}

def one_hot(value: str, domain: tuple[str, ...]) -> list[int]:
    """
    Deterministic one-hot encoder that never "fits" at runtime.
    Unknown values go to all-zeros vector.
    """
    if domain not in _CACHED_OHE:
        _CACHED_OHE[domain] = {v: idx for idx, v in enumerate(domain)}
    vec = [0] * len(domain)
    if value in _CACHED_OHE[domain]:
        vec[_CACHED_OHE[domain][value]] = 1
    return vec

def parse_json_field(json_str: str, default=None):
    """
    Parse a JSON string field into a Python object.
    Returns the default value if parsing fails.
    """
    if not json_str or not isinstance(json_str, str):
        return default
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default 
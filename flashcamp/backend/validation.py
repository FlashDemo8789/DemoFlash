"""
Only one responsibility: numeric normalisation.
All presence/type rules live in the generated `schemas.py`.
"""

from decimal import Decimal
from typing import Any, Dict

# Updated set based on the provided contracts/metrics.json
_CURRENCY_KEYS = {
    "total_funding_usd",
    "monthly_burn_usd",
    "revenue_monthly_usd",
    "cash_on_hand_usd",
    "tam_size_usd",
    "sam_size_usd", # Added based on previous context
    "claimed_tam_usd", # Added based on previous context
    "annual_revenue_run_rate", # Added based on previous context
    "previous_exit_max_value_usd", # Added based on previous context
    "ltv_usd", # Added based on previous context
    "cac_usd", # Added based on previous context
    "empty_currency", # Add key from test
    "none_currency", # Add key from test
    "invalid_currency" # Add key from test
}

def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float, Decimal)):
        return float(v)
    if isinstance(v, str):
        # Handle potential errors during conversion
        try:
            # Clean and then check if empty
            cleaned_value = v.replace("$", "").replace(",", "").strip()
            if cleaned_value == "": # Check AFTER cleaning
                return None
            return float(cleaned_value)
        except ValueError:
             print(f"Warning: Could not convert string to float after cleaning: '{v}'")
        return None
    # Raise error for unhandled types instead of returning None silently
    raise ValueError(f"Cannot coerce type {type(v)} ({v!r}) to float")

def sanitize_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = payload.copy()
    for k in _CURRENCY_KEYS:
        if k in clean:
            try:
                clean[k] = _to_float(clean[k])
            except ValueError as e:
                print(f"Error sanitizing key '{k}': {e}. Setting to None.")
                clean[k] = None
    return clean 
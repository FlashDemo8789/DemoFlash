from importlib import import_module

_modules = {
    "C_efficiency": "capital_efficiency",
    "A_moat":       "competitive_moat",
    "M_tam":        "tam_reasonableness",
    "P_team":       "team_quality",
    "C_runway":     "runway_mc",
    "M_competition": "competition_intensity",
    "M_reg_risk":    "reg_macro_risk",
}

analyses = {k: import_module(f"flashcamp.analysis.{m}").score
            for k, m in _modules.items()}

PILLAR_MAP = {
    "capital":   ["C_efficiency", "C_runway"],
    "advantage": ["A_moat"],
    "market":    ["M_tam", "M_competition", "M_reg_risk"],
    "people":    ["P_team"],
}

def pillar_scores(payload: dict) -> dict:
    scores = {}
    for pillar, analysis_keys in PILLAR_MAP.items():
        pillar_total_score = 0
        count = 0
        for key in analysis_keys:
            # Handle potential non-numeric scores like from TAM
            try:
                score_val = analyses[key](payload)
                if isinstance(score_val, (int, float)):
                     pillar_total_score += score_val
                     count += 1
                else:
                    # Decide how to handle non-numeric scores (e.g., TAM 'Green'/'Amber'/'Red')
                    # Option 1: Assign a numeric value (e.g., Green=100, Amber=50, Red=0)
                    # Option 2: Exclude them from numeric pillar average
                    # Option 3: Return them separately
                    # For now, excluding from average:
                    print(f"Note: Non-numeric score for {key} excluded from {pillar} pillar average.")
                    # Optionally return these structured scores separately if needed
            except Exception as e:
                 print(f"Error running analysis {key} for pillar {pillar}: {e}")

        # Calculate average score for the pillar if numeric scores were found
        scores[pillar] = round(pillar_total_score / count, 2) if count > 0 else 0 # Avoid division by zero

    return scores

# ------------------------------------------------------------------
# Cross-pillar imbalance detector
# ------------------------------------------------------------------
def imbalance_alert(pillars: dict) -> str | None:
    """
    If the spread between highest and lowest numeric pillar scores exceeds
    25 points, return an alert string naming the weakest pillar.
    """
    def _to_num(val):
        if isinstance(val, (int, float)):
            return val
        # map traffic-light strings from TAM or Risk modules
        cmap = {"Green": 80, "Light-green": 70, "Amber": 50,
                "Red": 30, "Deep-red": 20}
        return cmap.get(val, 50) # Default to 50 if not a recognized string

    # Ensure there are pillars to compare
    if not pillars:
        return None
        
    numeric = {k: _to_num(v) for k, v in pillars.items()}
    
    # Ensure there are numeric scores to compare
    if not numeric:
        return None
        
    hi_key = max(numeric, key=numeric.get)
    lo_key = min(numeric, key=numeric.get)
    
    hi_val = numeric[hi_key]
    lo_val = numeric[lo_key]

    # Use round to avoid float precision issues in comparison
    if round(hi_val - lo_val, 2) > 25:
        # Format values to 1 decimal place for the message
        hi_val_fmt = f"{hi_val:.1f}"
        lo_val_fmt = f"{lo_val:.1f}"
        return (
            f"Pillar imbalance detected: **{hi_key.capitalize()} {hi_val_fmt} "
            f"vs {lo_key.capitalize()} {lo_val_fmt}** – focus on {lo_key}."
        )
    return None

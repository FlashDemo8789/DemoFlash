"""
Regulatory & Macro Risk (M-3)
─────────────────────────────
Rescales a composite risk index (0–100) to a 0–10 score for the Market
pillar.  Lower risk → higher score.

Input column (already in the 99-col contract)
---------------------------------------------
regulatory_risk_score : float   # 0 (no risk) … 100 (extreme)

Mapping
-------
0-10   → score 10 (Green)
10-30  → score 8  (Light-green)
30-60  → score 5  (Amber)
60-80  → score 2  (Red)
>80    → score 0  (Deep-red)
"""

from .utils import get

THRESHOLDS = [
    (10, 10.0, "Green"),
    (30,  8.0, "Light-green"),
    (60,  5.0, "Amber"),
    (80,  2.0, "Red"),
    (101, 0.0, "Deep-red"), # Use 101 to capture >= 80
]


def score(payload: dict) -> float:
    risk = float(get(payload, "regulatory_risk_score"))
    for thresh, numeric, label in THRESHOLDS:
        if risk < thresh:
            payload["_reg_risk_bucket"] = label  # attach for PDF/UI
            return numeric
    # Fallback should theoretically not be reached if input is <= 100
    # but included for safety
    payload["_reg_risk_bucket"] = "Deep-red"
    return 0.0 
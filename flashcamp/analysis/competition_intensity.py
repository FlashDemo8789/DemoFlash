"""
Competition Intensity (M-2)
───────────────────────────
Maps market concentration to a categorical risk bucket and a numeric
score that feeds the Market pillar.

Input
-----
competition_intensity : float   # Herfindahl-proxy 0–1

Heuristic thresholds  (aligned with antitrust literature)
---------------------------------------------------------
H < 0.15   →  Low  competition   → score 90
0.15-0.25  →  Medium competition → score 60
H > 0.25   →  High competition   → score 30
"""

from .utils import get


BUCKETS = [
    (0.15, "Low",    90.0),
    (0.25, "Medium", 60.0),
    (1.00, "High",   30.0),
]


def score(payload: dict) -> float:
    h = float(get(payload, "competition_intensity"))
    # iterate bucket list until threshold exceeded
    for thresh, label, numeric in BUCKETS:
        if h < thresh:
            payload["_competition_bucket"] = label  # for PDF / UI tooltip
            return numeric
    # fallback (should never hit because last thresh=1.0)
    payload["_competition_bucket"] = "High"
    return 30.0 
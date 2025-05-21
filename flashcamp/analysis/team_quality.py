from .utils import get
import math

BETA0 = -1.2
WEIGHTS = {
    "years_experience_avg": 0.05,
    "prior_successful_exits_count": 0.8,
    "board_advisor_experience_score": 0.02,
}

def _sigmoid(z): return 1 / (1 + math.exp(-z))

def score(d: dict) -> float:
    z = BETA0 + sum(get(d, k) * w for k, w in WEIGHTS.items())
    return round(_sigmoid(z) * 100, 1)            # 0-100 
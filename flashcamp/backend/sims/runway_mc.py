import numpy as np
from typing import Dict, Any

def simulate_runway(d: Dict[str, Any], horizon_months: int = 36, sims: int = 10_000):
    cash = d["cash_on_hand_usd"] + d.get("committed_funding_usd", 0)
    burn_mu = d["monthly_burn_usd"]
    burn_sigma = d.get("burn_stddev_usd", burn_mu * 0.1)

    draws = np.random.normal(burn_mu, burn_sigma, size=(sims, horizon_months)).clip(min=1)
    cumulative = draws.cumsum(axis=1)
    bankrupt = (cumulative >= cash)
    first_short = np.where(bankrupt, np.argmax(bankrupt, axis=1) + 1, horizon_months + 1)
    prob_short_12 = (first_short <= 12).mean()
    median_runway = np.median(first_short)
    return {
        "prob_runway_lt_12m": float(prob_short_12),
        "median_runway_months": int(median_runway),
        "first_short_months": first_short.tolist()
    } 
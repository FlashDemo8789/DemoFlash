"""
Runway Monte-Carlo (C-1)
────────────────────────
Estimates how long a startup's cash lasts under burn-rate volatility.

Inputs  (all exist in the 99-col contract)
-------
cash_on_hand_usd          – current cash balance
burn_rate_usd             – mean monthly burn
burn_stddev_usd           – monthly burn σ  (0 → deterministic)
committed_funding_usd     – signed term-sheets not yet received
sim_count                 – optional override in payload (default 10 000)

Outputs
-------
dict(
    median_runway_months = float,
    prob_lt_12_months    = float   # [0-1]
)

The pillar numeric score returned to the backend is
100 × (1 – prob_lt_12_months).
"""

from __future__ import annotations
import numpy as np
from .utils import get


def _simulate_runway(
    cash: float, burn_mean: float, burn_std: float, committed: float, n: int = 10_000
) -> np.ndarray:
    """Return an array of runway months for n trials."""
    if burn_mean <= 0:
        return np.full(n, np.inf)  # negative burn => infinite runway
    # Ensure burn_std is non-negative
    burn_std = max(0, burn_std)
    burns = np.random.normal(loc=burn_mean, scale=burn_std, size=n).clip(min=1)
    return (cash + committed) / burns


def score(payload: dict) -> float:
    """
    Compute Capital pillar contribution (0-100) from runway probability.
    Also returns granular stats for report_service if caller requests it.
    """
    cash      = get(payload, "cash_on_hand_usd")
    burn_m    = get(payload, "burn_rate_usd")
    burn_sd   = get(payload, "burn_stddev_usd")
    committed = get(payload, "committed_funding_usd")
    sims      = int(payload.get("sim_count", 10_000))

    runs = _simulate_runway(cash, burn_m, burn_sd, committed, sims)
    prob_lt_12 = float((runs < 12).mean())
    median_run = float(np.median(runs))

    # attach granular metrics for downstream PDF/report
    # Use a consistent key, perhaps prefixed to avoid clashes
    payload["_analysis_runway_stats"] = {
        "median_runway_months": round(median_run, 1) if np.isfinite(median_run) else None, # Handle inf case
        "prob_lt_12_months": round(prob_lt_12, 3),
    }

    return round(100 * (1 - prob_lt_12), 1)  # higher is better 
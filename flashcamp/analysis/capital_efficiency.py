from .utils import get

def score(d: dict) -> float:
    """Capital-Efficiency 0-100."""
    burn = get(d, "burn_rate_usd") * 12              # annual burn
    arr  = max(get(d, "annual_revenue_run_rate"), 1) # avoid /0
    burn_multiple = burn / arr                       # <1 is great

    ltv = get(d, "customer_lifetime_value_usd")
    cac = max(get(d, "customer_acquisition_cost_usd"), 1)
    ltv_cac = ltv / cac                              # >3 is great

    burn_score = max(0, 1 - min(burn_multiple / 3, 1))   # 0–1
    ltv_score  = min(ltv_cac / 3, 1)                     # 0–1
    return round(100 * (0.6 * burn_score + 0.4 * ltv_score), 1) 
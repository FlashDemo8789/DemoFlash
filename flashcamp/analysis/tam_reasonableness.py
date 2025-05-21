from .utils import get
BASELINE_TAM = {"Fintech": 2e12, "Healthtech": 4e12, "__default__": 1e12}

def score(d: dict) -> str:
    claimed  = get(d, "tam_size_usd")
    industry = get(d, "industry", "__default__")
    baseline = BASELINE_TAM.get(industry, BASELINE_TAM["__default__"])
    ratio    = claimed / baseline if baseline else 0
    ok_growth = get(d, "market_growth_rate_percent") <= 50
    within   = 0.3 <= ratio <= 3 and ok_growth
    return "Green" if within else ("Amber" if ratio < 0.1 or ratio > 5 else "Red") 
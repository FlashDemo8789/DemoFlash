from .utils import get

SWITCH_IDX = {0: 0.0, 1: 0.5, 2: 1.0}  # Low/Med/High already numeric

def score(d: dict) -> float:
    patents  = min(get(d, "patent_count"), 10) / 10
    network  = 1.0 if get(d, "network_effects_present") else 0.0
    switch   = SWITCH_IDX.get(get(d, "switching_cost_score", 0), 0)
    moat     = 0.4 * patents + 0.4 * network + 0.2 * switch
    return round(10 * moat, 2)                # 0-10 rating 
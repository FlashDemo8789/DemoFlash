import numpy as np
from typing import Sequence, Dict

def portfolio_summary(success_probs: Sequence[float],
                      cheque_sizes: Sequence[float],
                      sims: int = 50_000) -> Dict:
    p = np.asarray(success_probs)
    cheques = np.asarray(cheque_sizes)
    exits = np.random.binomial(1, p, size=(sims, len(p)))
    returns = (exits * cheques * 10)  # assume 10× on success, 0 on fail
    portfolio_irrs = returns.sum(axis=1) / cheques.sum() - 1
    return {
        "irr_p50": float(np.percentile(portfolio_irrs, 50) * 100),
        "irr_p10": float(np.percentile(portfolio_irrs, 10) * 100),
        "irr_p90": float(np.percentile(portfolio_irrs, 90) * 100)
    } 
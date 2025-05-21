import json, pathlib, pytest
from flashcamp.backend.schemas import MetricsInput

SPEC = json.loads((pathlib.Path(__file__).parents[1] / "contracts/metrics.json").read_text())
def minimal(): return {k: v["example"] for k, v in SPEC.items() if v["required"]}

def test_required_enforced():
    bad = minimal(); bad.pop("startup_id")
    with pytest.raises(ValueError): MetricsInput(**bad)

def test_runway_invariant():
    bad = minimal()
    bad.update(cash_on_hand_usd=120000, monthly_burn_usd=6000, runway_months=10)
    with pytest.raises(ValueError): MetricsInput(**bad)

def test_forbid_extra():
    bad = minimal(); bad["unexpected"] = 1
    with pytest.raises(ValueError): MetricsInput(**bad) 
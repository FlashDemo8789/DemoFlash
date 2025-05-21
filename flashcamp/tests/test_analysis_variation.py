import json, pathlib
import pytest
from fastapi.testclient import TestClient
from flashcamp import app

SPEC = json.loads((pathlib.Path(__file__).parents[1] / "contracts/metrics.json").read_text())
client = TestClient(app)

def _payload(**overrides):
    base = {k: v["example"] for k, v in SPEC.items() if v["required"]}
    base.update(overrides)
    return base

@pytest.mark.parametrize("market_growth", [5, 15, 30])
def test_success_prob_varies_with_market_growth(market_growth):
    body = _payload(market_growth_rate_percent=market_growth)
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200
    assert 0.0 <= r.json()["success_probability"] <= 1.0 
"""The model-call estimate must track the phases the orchestrator actually runs."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.estimate import estimate_calls
from app.main import app
from app.schemas import CaseInput


def _case(**kw) -> CaseInput:
    base = dict(title="t", charge_or_claim="c", your_side="s")
    base.update(kw)
    return CaseInput(**base)


def test_default_configuration_is_pinned():
    # Hand-counted against orchestrator.run_trial for the defaults (jury 3, 2
    # argument rounds, 2 deliberation rounds in dialogue style, personas, digest,
    # straw poll, exhortation, 1 pass, no witnesses, no grounding):
    #   setup 8 + arguments 5 + closings 2 + digest 1 + straw 3
    #   + deliberation (1 + 3 + 3) = 7 per round -> min 7, max 3 rounds x 7 + 1
    #   + ruling 1
    est = estimate_calls(_case())
    assert est["min_calls"] == 27
    assert est["max_calls"] == 42
    assert est["breakdown"]["deliberation"] == {"min": 7, "max": 22}


def test_passes_multiply_only_the_ballots():
    one = estimate_calls(_case(verdict_passes=1))
    three = estimate_calls(_case(verdict_passes=3))
    # 3 jurors x 2 extra samples = 6 more calls per deliberation round.
    assert three["breakdown"]["deliberation"]["min"] - one["breakdown"]["deliberation"]["min"] == 6
    assert three["breakdown"]["setup"] == one["breakdown"]["setup"]


def test_witnesses_add_three_to_four_calls_per_exchange():
    w = [{"name": "W", "role": "expert", "called_by": "prosecution", "what_they_know": "x"}]
    est = estimate_calls(_case(witnesses=w, qa_exchanges=2, qa_redirect=1))
    # direct 2 + cross 2 + redirect 1 = 5 exchanges
    assert est["breakdown"]["evidence"] == {"min": 15, "max": 20}
    assert est["breakdown"]["directed_verdict"] == {"min": 1, "max": 2}
    assert "evidence" not in estimate_calls(_case())["breakdown"]


def test_poll_style_and_no_exhortation_are_cheaper():
    dialogue = estimate_calls(_case())
    poll = estimate_calls(_case(deliberation_style="poll"))
    assert poll["max_calls"] < dialogue["max_calls"]
    no_exhort = estimate_calls(_case(deadlock_exhortation=False))
    assert no_exhort["max_calls"] < dialogue["max_calls"]
    assert no_exhort["min_calls"] == dialogue["min_calls"]  # the floor never used the extra round


def test_grounding_adds_checkers():
    off = estimate_calls(_case())
    on = estimate_calls(_case(grounding_check=True))  # closing + ruling by default
    assert on["breakdown"]["closings"]["min"] == off["breakdown"]["closings"]["min"] + 2
    assert on["breakdown"]["ruling"]["min"] == off["breakdown"]["ruling"]["min"] + 1
    adversarial = estimate_calls(_case(grounding_check=True, grounding_adversarial=True))
    assert adversarial["breakdown"]["ruling"]["min"] == off["breakdown"]["ruling"]["min"] + 3


def test_estimate_endpoint_returns_the_range():
    client = TestClient(app)
    r = client.post("/api/estimate", json={"title": "t", "charge_or_claim": "c", "your_side": "s"})
    assert r.status_code == 200
    body = r.json()
    assert body["min_calls"] == 27 and body["max_calls"] == 42


def test_estimate_endpoint_rejects_an_invalid_case():
    client = TestClient(app)
    assert client.post("/api/estimate", json={"title": ""}).status_code == 422

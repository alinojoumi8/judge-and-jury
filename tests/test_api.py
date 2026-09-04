"""The HTTP surface: rate limiting, the concurrency cap, and the optional API key.

`run_trial` is replaced with a two-event stub so nothing here touches a model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ratelimit import ConcurrencyGate, TokenBucket
from app.schemas import TrialEvent

CASE = {"title": "t", "charge_or_claim": "c", "your_side": "s"}


async def _stub_run_trial(case):
    yield TrialEvent(phase="Intake", speaker="System", kind="phase", content="Case Intake")
    yield TrialEvent(phase="done", speaker="System", kind="done")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "run_trial", _stub_run_trial)
    monkeypatch.setattr(main, "_bucket", TokenBucket())
    monkeypatch.setattr(main, "_gate", ConcurrencyGate())
    monkeypatch.delenv("COURTROOM_API_KEY", raising=False)
    monkeypatch.delenv("TRIAL_RATE_PER_MINUTE", raising=False)
    monkeypatch.delenv("TRIAL_MAX_CONCURRENT", raising=False)
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# Pure limiters
# ---------------------------------------------------------------------------
def test_token_bucket_spends_then_refills():
    now = [1000.0]
    bucket = TokenBucket(clock=lambda: now[0])
    assert bucket.allow("a", 2)[0] and bucket.allow("a", 2)[0]
    refused, wait = bucket.allow("a", 2)
    assert refused is False and 29.0 < wait <= 30.0  # one token per 30s at 2/min
    now[0] += 30.0
    assert bucket.allow("a", 2)[0]
    assert bucket.allow("b", 2)[0]  # a different client has its own bucket


def test_token_bucket_zero_rate_means_unlimited():
    bucket = TokenBucket(clock=lambda: 0.0)
    assert all(bucket.allow("a", 0)[0] for _ in range(50))


def test_concurrency_gate_caps_and_releases():
    gate = ConcurrencyGate()
    assert gate.try_acquire(1) is True
    assert gate.try_acquire(1) is False
    gate.release()
    assert gate.try_acquire(1) is True
    gate.release(); gate.release()  # over-release never goes negative
    assert gate.active == 0


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------
def test_trial_streams_by_default(client):
    r = client.post("/api/trial", json=CASE)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert '"kind":"done"' in r.text.replace(" ", "")


def test_burst_is_rate_limited_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("TRIAL_RATE_PER_MINUTE", "2")
    assert client.post("/api/trial", json=CASE).status_code == 200
    assert client.post("/api/trial", json=CASE).status_code == 200
    r = client.post("/api/trial", json=CASE)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert "retry_after" in r.json()


def test_gate_slot_is_released_after_a_stream(client, monkeypatch):
    monkeypatch.setenv("TRIAL_MAX_CONCURRENT", "1")
    for _ in range(3):  # each stream fully consumed -> slot released -> next allowed
        assert client.post("/api/trial", json=CASE).status_code == 200
    assert main._gate.active == 0


def test_gate_refuses_when_full(client, monkeypatch):
    monkeypatch.setenv("TRIAL_MAX_CONCURRENT", "1")
    main._gate.try_acquire(1)  # a trial already in flight
    r = client.post("/api/trial", json=CASE)
    assert r.status_code == 429 and "in progress" in r.json()["error"]


def test_api_key_gates_trial_and_estimate_when_configured(client, monkeypatch):
    monkeypatch.setenv("COURTROOM_API_KEY", "s3cret")
    assert client.post("/api/trial", json=CASE).status_code == 401
    assert client.post("/api/trial", json=CASE, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/estimate", json=CASE).status_code == 401
    assert client.post("/api/trial", json=CASE, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.post("/api/estimate", json=CASE, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_health_reports_whether_a_key_is_required(client, monkeypatch):
    assert client.get("/api/health").json()["auth_required"] is False
    monkeypatch.setenv("COURTROOM_API_KEY", "s3cret")
    assert client.get("/api/health").json()["auth_required"] is True  # readable without the key


def test_no_key_configured_means_open_access(client):
    assert client.post("/api/estimate", json=CASE).status_code == 200

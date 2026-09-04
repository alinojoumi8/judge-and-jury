"""Test-wide guards.

Nothing in the suite may depend on — or be able to use — a real model key. The
developer machine has a `.env` that `app.config` loads at import; CI has none.
Blanking the key for every test makes the suite behave here exactly as it does
there, so a test that quietly reaches for settings fails on the laptop first
instead of on the build (that is how the offline trial tests once went red only
in CI).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_model_key(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "")

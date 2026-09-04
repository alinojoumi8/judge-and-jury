"""FastAPI app: serves the courtroom UI and streams trials over SSE."""

from __future__ import annotations

import hmac
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .estimate import estimate_calls
from .orchestrator import run_trial
from .ratelimit import ConcurrencyGate, TokenBucket
from .schemas import CaseInput

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="AI Courtroom — Judge & Jury")

# Every trial costs real money, and /api/trial is reachable by anyone who can reach
# the server. On localhost the defaults change nothing you would notice; the moment
# the server is exposed they are what stands between a stray request and a bill.
# Limits are read from the environment per request so they can be changed (and
# tested) without restarting.
_bucket = TokenBucket()
_gate = ConcurrencyGate()


def _rate_per_minute() -> float:
    """Trials each client may start per minute (0 disables the limit)."""
    return float(os.getenv("TRIAL_RATE_PER_MINUTE", "6") or 6)


def _max_concurrent() -> int:
    """Trials that may be in flight at once, across all clients."""
    return max(1, int(os.getenv("TRIAL_MAX_CONCURRENT", "2") or 2))


def _api_key() -> str:
    """When set, every trial and estimate request must carry it in X-API-Key."""
    return os.getenv("COURTROOM_API_KEY", "").strip()


def _require_key(request: Request) -> None:
    expected = _api_key()
    if not expected:
        return
    given = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")


def _client_key(request: Request) -> str:
    # The direct peer. Behind a reverse proxy this is the proxy: pass the real
    # client address through if you deploy that way, or rate-limit at the proxy.
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def fresh_static_assets(request: Request, call_next):
    """Revalidate static assets so edits to the UI always show up (no stale cache)."""
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> JSONResponse:
    """Confirm a MiniMax key/model is configured (without calling the API)."""
    auth = bool(_api_key())
    try:
        s = get_settings()
        return JSONResponse(
            {"ok": True, "model": s.model, "base_url": s.base_url, "auth_required": auth}
        )
    except Exception as exc:  # missing key, etc.
        return JSONResponse({"ok": False, "error": str(exc), "auth_required": auth}, status_code=503)


@app.post("/api/estimate")
async def estimate(case: CaseInput, request: Request) -> JSONResponse:
    """How many model calls this configuration will cost, before anything runs."""
    _require_key(request)
    return JSONResponse(estimate_calls(case))


@app.post("/api/trial")
async def trial(case: CaseInput, request: Request):
    _require_key(request)
    allowed, wait = _bucket.allow(_client_key(request), _rate_per_minute())
    if not allowed:
        retry = max(1, math.ceil(wait))
        return JSONResponse(
            {"error": f"Too many trials started recently; try again in {retry}s.", "retry_after": retry},
            status_code=429,
            headers={"Retry-After": str(retry)},
        )
    if not _gate.try_acquire(_max_concurrent()):
        return JSONResponse(
            {"error": "The maximum number of trials is already in progress; try again shortly.",
             "retry_after": 30},
            status_code=429,
            headers={"Retry-After": "30"},
        )

    async def event_stream():
        try:
            async for ev in run_trial(case):
                yield f"data: {ev.model_dump_json()}\n\n"
        finally:
            # Runs on normal completion, on error, and on the cancellation a client
            # disconnect triggers — the slot is never leaked.
            _gate.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so deltas flush live
        },
    )


# Serve static assets (styles.css, app.js) from /static.
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

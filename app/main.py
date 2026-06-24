"""FastAPI app: serves the courtroom UI and streams trials over SSE."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .orchestrator import run_trial
from .schemas import CaseInput

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="AI Courtroom — Judge & Jury")


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
    try:
        s = get_settings()
        return JSONResponse({"ok": True, "model": s.model, "base_url": s.base_url})
    except Exception as exc:  # missing key, etc.
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.post("/api/trial")
async def trial(case: CaseInput) -> StreamingResponse:
    async def event_stream():
        async for ev in run_trial(case):
            yield f"data: {ev.model_dump_json()}\n\n"

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

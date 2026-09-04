"""Helpers for working with MiniMax reasoning models through PydanticAI.

MiniMax M2.x models are *reasoning* models: they emit a `<think>...</think>`
block before their actual answer, and PydanticAI 2.0's tool/native structured
output doesn't reliably extract data from them. So we run every agent in plain
text mode, ask it for JSON, then strip the think block and parse + validate the
JSON ourselves (see `run_structured`). Every role — including the speaking ones —
returns JSON, which keeps the full statement intact (the model otherwise leaks
the opening words of a statement into its think block).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Control characters that JSON forbids *unescaped* inside string values. MiniMax
# often emits literal newlines/tabs inside its "statement" value, which the strict
# JSON parser rejects — so we parse leniently and escape stray control chars.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_TO_END = re.compile(r"<think>.*$", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)

# Ceiling on simultaneous model calls. Deliberation fans out one call per juror
# per sample — 12 jurors x 5 passes is 60 at once — and a burst like that turns
# into 429s. The client retries those a couple of times itself, but a 429 that
# outlives its retries surfaces here as a failed ballot, i.e. an abstention. A
# modest ceiling costs a little wall-clock and buys a full jury.
_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "8") or 8))
# First backoff after a failed model call; doubles per attempt.
_BACKOFF_SECONDS = max(0.0, float(os.getenv("LLM_RETRY_BACKOFF", "1.5") or 1.5))
# One semaphore per event loop: an asyncio.Semaphore binds to the loop it first
# waits on, and scripts (and tests) create a fresh loop per asyncio.run().
_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(_MAX_CONCURRENCY)
    return sem


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` (and any unclosed trailing think) from text."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN_TO_END.sub("", text)
    return text.strip()


def json_candidates(text: str) -> list[str]:
    """Every plausible JSON span in a reply, most likely first.

    A fenced block wins outright. Otherwise the outermost {...} is offered before
    the outermost [...]: picking whichever bracket appears first mis-reads a reply
    like 'Note [1]: {"a": 1}' as '[1]: {"a": 1}' and burns a retry on it, and every
    role in this app returns an object. The caller validates each candidate in
    turn, so a wrong guess costs nothing.
    """
    text = strip_think(text)
    fence = _JSON_FENCE.search(text)
    if fence:
        return [fence.group(1)]
    out: list[str] = []
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = text.find(open_), text.rfind(close)
        if start != -1 and end > start:
            out.append(text[start : end + 1])
    return out or [text]


def extract_json(text: str) -> str:
    """The single most likely JSON span in a reply (see `json_candidates`)."""
    return json_candidates(text)[0]


def parse_json_lenient(candidate: str) -> dict:
    """Parse model JSON, tolerating literal control chars inside string values.

    `json.loads(strict=False)` already allows literal newlines/tabs in strings;
    if anything else trips it up, we escape stray control characters and retry.
    """
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        escaped = _CTRL.sub(lambda m: f"\\u{ord(m.group()):04x}", candidate)
        return json.loads(escaped, strict=False)


def contains_cjk(text: str) -> bool:
    """True if `text` contains CJK / fullwidth characters.

    MiniMax reasoning models occasionally code-switch out of English mid-answer.
    A "write in English" instruction alone doesn't reliably prevent it, so we
    detect the slip and re-prompt (see `run_structured(require_english=True)`).
    """
    for ch in text:
        o = ord(ch)
        if (
            0x3000 <= o <= 0x303F      # CJK symbols & punctuation
            or 0x3400 <= o <= 0x9FFF   # CJK ideographs (ext-A + unified)
            or 0xF900 <= o <= 0xFAFF   # CJK compatibility ideographs
            or 0xFF00 <= o <= 0xFFEF   # halfwidth/fullwidth forms
        ):
            return True
    return False


def _model_text(obj: BaseModel) -> str:
    """Concatenate every string value in a (possibly nested) model — for scanning."""
    parts: list[str] = []

    def walk(v: object) -> None:
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, BaseModel):
            for x in v.__dict__.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(obj)
    return " ".join(parts)


T = TypeVar("T", bound=BaseModel)


async def run_structured(
    agent,
    prompt: str,
    model_cls: Type[T],
    retries: int = 2,
    *,
    require_english: bool = False,
) -> T:
    """Run a plain-text agent and parse/validate its reply into `model_cls`.

    One retry budget covers three kinds of failure: the model call itself (a
    timeout, or a 429 that outlived the client's own retries — backed off and
    tried again), a reply that does not parse or validate (re-prompted with the
    error), and, with `require_english=True`, a reply that slipped into CJK
    (re-prompted in English).
    """
    last_err = ""
    retry_note = ""
    for attempt in range(retries + 1):
        p = f"{prompt}\n\n{retry_note}" if retry_note else prompt
        try:
            async with _semaphore():
                result = await agent.run(p)
        except Exception as exc:  # transport / provider failure, not a bad reply
            last_err = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning(
                "Model call failed (attempt %d/%d): %s", attempt + 1, retries + 1, last_err
            )
            if attempt < retries:
                await asyncio.sleep(_BACKOFF_SECONDS * (2 ** attempt))
            continue
        raw = result.output if isinstance(result.output, str) else str(result.output)
        obj: T | None = None
        for candidate in json_candidates(raw):
            try:
                obj = model_cls.model_validate(parse_json_lenient(candidate))
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)[:200]
        if obj is None:
            retry_note = (
                "Your previous reply could not be parsed as JSON for the required "
                f"schema (error: {last_err}). Reply with ONLY a single valid JSON "
                "object — no prose, no markdown fences."
            )
            continue
        if require_english and contains_cjk(_model_text(obj)):
            last_err = "non-English (CJK) characters in output"
            retry_note = (
                "Your previous reply contained non-English characters. Reply again "
                "using the SAME JSON schema but written ENTIRELY IN ENGLISH — do not "
                "use any Chinese or other non-Latin characters."
            )
            continue
        return obj
    raise ValueError(
        f"Could not produce a valid {model_cls.__name__} after {retries + 1} tries: {last_err}"
    )

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

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel

# Control characters that JSON forbids *unescaped* inside string values. MiniMax
# often emits literal newlines/tabs inside its "statement" value, which the strict
# JSON parser rejects — so we parse leniently and escape stray control chars.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_TO_END = re.compile(r"<think>.*$", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` (and any unclosed trailing think) from text."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN_TO_END.sub("", text)
    return text.strip()


def extract_json(text: str) -> str:
    """Pull the JSON object/array out of a model reply (after stripping think)."""
    text = strip_think(text)
    fence = _JSON_FENCE.search(text)
    if fence:
        return fence.group(1)
    # Otherwise grab the outermost {...} or [...].
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if starts and ends:
        start, end = min(starts), max(ends)
        if end > start:
            return text[start : end + 1]
    return text


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


T = TypeVar("T", bound=BaseModel)


async def run_structured(agent, prompt: str, model_cls: Type[T], retries: int = 2) -> T:
    """Run a plain-text agent and parse/validate its reply into `model_cls`."""
    last_err = ""
    for attempt in range(retries + 1):
        p = prompt
        if attempt > 0:
            p = (
                f"{prompt}\n\nYour previous reply could not be parsed as JSON for the "
                f"required schema (error: {last_err}). Reply with ONLY a single valid "
                "JSON object — no prose, no markdown fences."
            )
        result = await agent.run(p)
        raw = result.output if isinstance(result.output, str) else str(result.output)
        candidate = extract_json(raw)
        try:
            return model_cls.model_validate(parse_json_lenient(candidate))
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)[:200]
    raise ValueError(f"Could not parse {model_cls.__name__} after {retries + 1} tries: {last_err}")

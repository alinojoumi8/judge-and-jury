"""Builds a MiniMax-backed PydanticAI model via the OpenAI-compatible endpoint."""

from __future__ import annotations

from functools import lru_cache

# `OpenAIChatModel` is the current name; older PydanticAI exposed `OpenAIModel`.
try:  # pragma: no cover - import shim
    from pydantic_ai.models.openai import OpenAIChatModel as _OpenAIChatModel
except ImportError:  # pragma: no cover
    from pydantic_ai.models.openai import OpenAIModel as _OpenAIChatModel

from pydantic_ai.providers.openai import OpenAIProvider

from .config import get_settings


@lru_cache(maxsize=8)
def build_model(model_name: str | None = None):
    """Return a PydanticAI model pointed at MiniMax.

    MiniMax exposes a standard OpenAI-compatible endpoint, so we can drive it
    through PydanticAI's OpenAI model with a custom base_url + api_key.

    `model_name` optionally overrides the configured default (e.g. per case).
    """
    settings = get_settings()
    return _OpenAIChatModel(
        model_name or settings.model,
        provider=OpenAIProvider(base_url=settings.base_url, api_key=settings.api_key),
    )

"""Configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env (if present) once, when this module is first imported.
load_dotenv()

DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M2.1"


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str


def get_settings() -> Settings:
    """Read settings from the environment, validating that a key is present.

    Raised lazily (at trial time) rather than at import so the app/tests can be
    imported without a key configured.
    """
    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    base_url = os.getenv("MINIMAX_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    model = os.getenv("MINIMAX_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not api_key or api_key == "your-key-here":
        raise RuntimeError(
            "MINIMAX_API_KEY is not set. Copy .env.example to .env and paste your "
            "MiniMax token-plan key."
        )

    return Settings(api_key=api_key, base_url=base_url, model=model)

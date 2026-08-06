"""Courtroom role agents (PydanticAI), all powered by MiniMax."""

from .casting import build_caster_agent
from .crown import build_crown_agent
from .defense import build_defense_agent
from .digest import build_digest_agent
from .intake import build_intake_agent
from .judge import build_judge_agent, build_ruling_agent
from .juror import (
    build_deliberation_agent,
    build_foreperson_agent,
    build_juror_agent,
    build_juror_pool_agent,
)
from .strategy import build_strategist_agent
from .verifier import build_verifier_agent
from .witness import build_witness_agent

__all__ = [
    "build_intake_agent",
    "build_digest_agent",
    "build_crown_agent",
    "build_defense_agent",
    "build_judge_agent",
    "build_ruling_agent",
    "build_juror_agent",
    "build_juror_pool_agent",
    "build_deliberation_agent",
    "build_foreperson_agent",
    "build_strategist_agent",
    "build_verifier_agent",
    "build_witness_agent",
]

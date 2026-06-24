"""Pydantic models: case input, agent outputs, and the streamed trial events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CaseType = Literal["criminal", "civil"]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class CaseInput(BaseModel):
    """What the user submits from the intake form."""

    title: str = Field(..., min_length=1, description="Short name for the case.")
    case_type: CaseType = "criminal"
    jurisdiction: str = "Ontario, Canada"
    charge_or_claim: str = Field(
        ..., min_length=1, description="The charge (criminal) or claim (civil)."
    )
    your_side: str = Field(
        ..., min_length=1, description="The client's account / side of the story."
    )
    jury_size: int = Field(default=3, ge=1, le=12)
    argument_rounds: int = Field(default=2, ge=1, le=5)
    model: str | None = Field(default=None, description="Optional MiniMax model override.")


# ---------------------------------------------------------------------------
# Agent outputs (structured)
# ---------------------------------------------------------------------------
class Speech(BaseModel):
    """A spoken statement from a free-form role (Crown, Defense, Judge)."""

    statement: str


class StructuredCase(BaseModel):
    """Intake agent's structured view of the case.

    Fields default to empty so an occasional missing key from the model never
    crashes a whole trial.
    """

    case_caption: str = "The Case"
    charges_or_claims: list[str] = Field(default_factory=list)
    summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    prosecution_theory: str = ""
    defense_theory: str = ""


class JurorPersona(BaseModel):
    name: str = "Juror"
    background: str = ""
    disposition: str = ""


class JuryPool(BaseModel):
    jurors: list[JurorPersona] = Field(default_factory=list)


class JurorVote(BaseModel):
    juror_name: str = "Juror"
    verdict: str = "not guilty"  # "guilty"/"not guilty" or "liable"/"not liable"
    confidence: int = Field(default=5, ge=1, le=10)
    reasoning: str = ""


class Verdict(BaseModel):
    """Computed in code from the jury's votes."""

    tally: dict[str, int]
    outcome: str
    unanimous: bool
    hung: bool
    dissent_summary: str


class JudgeRuling(BaseModel):
    verdict_acknowledgement: str = ""
    reasoning: str = ""
    sentence_or_remedy: str = ""
    closing_remarks: str = ""


# ---------------------------------------------------------------------------
# Streaming unit
# ---------------------------------------------------------------------------
EventKind = Literal[
    "phase",          # a phase header / divider
    "speaker_start",  # a new speaker turn begins (free-form, streamed)
    "delta",          # incremental text appended to the current speaker bubble
    "message",        # a complete (non-streamed) text message
    "structured",     # a structured payload (data + optional summary content)
    "error",
    "done",
]


class TrialEvent(BaseModel):
    """One unit streamed to the browser over SSE."""

    phase: str
    speaker: str
    kind: EventKind
    content: str = ""
    data: dict[str, Any] | None = None

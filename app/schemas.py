"""Pydantic models: case input, agent outputs, and the streamed trial events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CaseType = Literal["criminal", "civil"]
# A juror's normalized decision. Free-text verdict wording is kept for display,
# but the tally is driven by this enum so we never have to guess from prose.
VoteChoice = Literal["convict", "acquit"]
WitnessRole = Literal[
    "complainant", "investigator", "expert", "character", "defense_witness", "other"
]


# ---------------------------------------------------------------------------
# Input sub-models
# ---------------------------------------------------------------------------
class Defendant(BaseModel):
    """One co-accused, judged on their own role and conduct."""

    name: str = Field(..., min_length=1)
    role: str = ""  # e.g. "CEO", "signatory director", "salesman"
    account: str = ""  # this defendant's individual side of the story


class Witness(BaseModel):
    """A witness who can be examined and cross-examined."""

    name: str = Field(..., min_length=1)
    role: WitnessRole = "other"
    called_by: Literal["prosecution", "defense"] = "prosecution"
    what_they_know: str = ""


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
    # How many times the jury votes, seeing each other's reasoning between rounds.
    deliberation_rounds: int = Field(default=2, ge=1, le=4)
    # Optional co-accused. Empty => single-accused trial (classic behaviour).
    defendants: list[Defendant] = Field(default_factory=list)
    # Optional witnesses. Empty => no evidence phase (classic behaviour).
    witnesses: list[Witness] = Field(default_factory=list)
    max_witnesses: int = Field(default=3, ge=1, le=5)
    qa_exchanges: int = Field(default=2, ge=1, le=4)
    model: str | None = Field(default=None, description="Optional MiniMax model override.")


# ---------------------------------------------------------------------------
# Agent outputs (structured)
# ---------------------------------------------------------------------------
class Speech(BaseModel):
    """A spoken statement from a free-form role (Crown, Defense, Judge, Witness)."""

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


class DefendantVote(BaseModel):
    """A juror's verdict for one co-accused (multi-defendant trials)."""

    defendant_name: str = ""
    verdict: str = "not guilty"
    vote: VoteChoice = "acquit"
    confidence: int = Field(default=5, ge=1, le=10)
    reasoning: str = ""


class JurorVote(BaseModel):
    juror_name: str = "Juror"
    verdict: str = "not guilty"  # "guilty"/"not guilty" or "liable"/"not liable"
    vote: VoteChoice = "acquit"  # normalized convict/acquit signal used for tallying
    confidence: int = Field(default=5, ge=1, le=10)
    reasoning: str = ""
    # Populated only in multi-defendant trials (one entry per co-accused).
    defendant_votes: list[DefendantVote] = Field(default_factory=list)


class DefendantVerdict(BaseModel):
    """The jury's verdict for one co-accused."""

    defendant_name: str
    role: str = ""
    tally: dict[str, int]
    outcome: str
    unanimous: bool
    hung: bool
    dissent_summary: str


class Verdict(BaseModel):
    """Computed in code from the jury's votes.

    The top-level fields describe the (single) accused, or — in a multi-defendant
    trial — mirror the first defendant so older single-accused consumers keep
    working. `per_defendant` carries the full per-accused breakdown.
    """

    tally: dict[str, int]
    outcome: str
    unanimous: bool
    hung: bool
    dissent_summary: str
    per_defendant: list[DefendantVerdict] = Field(default_factory=list)


class JudgeRuling(BaseModel):
    verdict_acknowledgement: str = ""
    reasoning: str = ""
    sentence_or_remedy: str = ""
    closing_remarks: str = ""


# ---------------------------------------------------------------------------
# Witness-examination outputs
# ---------------------------------------------------------------------------
class ExaminationQuestion(BaseModel):
    question: str


class WitnessAnswer(BaseModel):
    statement: str


class Objection(BaseModel):
    object: bool = False
    ground: str = ""  # "leading" | "hearsay" | "speculation" | "relevance"
    text: str = ""


class ObjectionRuling(BaseModel):
    ruling: str = "overruled"  # "sustained" | "overruled"
    text: str = ""


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

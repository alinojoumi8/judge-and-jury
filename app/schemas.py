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
# How the jury deliberates. "dialogue" = a sequential jury-room discussion where
# jurors hear and respond to each other before voting; "poll" = the classic
# parallel re-vote (cheaper, no cross-talk).
DeliberationStyle = Literal["dialogue", "poll"]
# Which phases the optional fact-check verifier scrutinises.
GroundingPhase = Literal["witness", "closing", "ruling"]
GroundingSeverity = Literal["minor", "moderate", "severe"]


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
    # Optional courtroom demeanour (nervous / confident / evasive / precise…). If
    # empty and personas are on, the casting director assigns one automatically.
    demeanour: str = ""


class RolePersona(BaseModel):
    """A distinct personality for a speaking role (counsel, judge, or witness).

    `style` is how they carry themselves: advocacy manner for counsel, bench
    temperament for the judge, courtroom demeanour for a witness. Used both as an
    auto-cast result and as an optional user-pinned input on CaseInput.
    """

    name: str = ""
    background: str = ""
    style: str = ""


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
    # Whether the jury holds a spoken deliberation (jurors respond to each other,
    # foreperson-led) or just re-polls. Defaults to the richer dialogue.
    deliberation_style: DeliberationStyle = "dialogue"
    # Take a private, independent straw poll BEFORE any discussion (anti-herding):
    # captures each juror's own first read before social influence, then shows how
    # the room moved. Fires only with more than one juror.
    straw_poll: bool = True
    # Auto-cast a distinct personality for the Crown, Defence, Judge, and each
    # witness (like the jury already gets). Off => generic role voices as before.
    personas: bool = True
    # Pin a specific personality for a role (overrides auto-casting that role) — e.g.
    # an aggressive senior prosecutor vs. a folksy defender, for A/B-ing how style
    # alone moves a verdict. A pinned role is honoured even when `personas` is off.
    crown_persona: RolePersona | None = None
    defense_persona: RolePersona | None = None
    judge_persona: RolePersona | None = None
    # Optional co-accused. Empty => single-accused trial (classic behaviour).
    defendants: list[Defendant] = Field(default_factory=list)
    # Optional witnesses. Empty => no evidence phase (classic behaviour).
    witnesses: list[Witness] = Field(default_factory=list)
    max_witnesses: int = Field(default=3, ge=1, le=5)
    qa_exchanges: int = Field(default=2, ge=1, le=4)
    # Re-examination after cross, and a defence directed-verdict motion (criminal
    # realism). Both fire only when there are witnesses; default on.
    redirect: bool = True
    qa_redirect: int = Field(default=1, ge=0, le=2)
    allow_directed_verdict: bool = True
    # Anti-hallucination fact-check pass (default OFF → identical behaviour/cost).
    grounding_check: bool = False
    grounding_phases: list[GroundingPhase] = Field(default_factory=lambda: ["closing", "ruling"])
    grounding_adversarial: bool = False   # run multiple checkers on the ruling and merge
    # Two-pass "draft → self-ground-check → revise" for witnesses/closings (opt-in).
    self_ground: bool = False
    model: str | None = Field(default=None, description="Optional MiniMax model override.")


# ---------------------------------------------------------------------------
# Agent outputs (structured)
# ---------------------------------------------------------------------------
class Speech(BaseModel):
    """A spoken statement from a free-form role (Crown, Defense, Judge, Witness)."""

    statement: str


class CaseStrategy(BaseModel):
    """One counsel's persistent theory of the case, carried across every phase.

    Built once after intake and threaded into that lawyer's opening, arguments and
    closing so their position stays coherent — and so they can directly steelman and
    rebut the opponent's single best point rather than re-deriving each turn.
    """

    theory: str = ""
    strongest_points: list[str] = Field(default_factory=list)
    opponents_best_point: str = ""
    rebuttal: str = ""


class AgreedRecord(BaseModel):
    """The immutable source-of-truth ledger, built once at intake.

    Threaded into every agent prompt so the whole trial is grounded: agents must
    argue ONLY from this Record plus on-record testimony, and must not invent
    facts, figures, dates, parties, or legal citations not listed here. Anything
    beyond the Record is argument/inference, never asserted fact.
    """

    parties: list[str] = Field(default_factory=list)        # canonical party names
    figures: list[str] = Field(default_factory=list)        # key numbers, each "label: value"
    dates: list[str] = Field(default_factory=list)          # key dates, each "label: date"
    admissible_facts: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)    # statutes/cases EXPLICITLY in the input


class Charge(BaseModel):
    """One charge / count, with its OWN essential elements.

    When a case has more than one charge, the jury returns a SEPARATE verdict on
    each (e.g. guilty of fraud, not guilty of possession).
    """

    label: str = ""              # e.g. "Fraud over $5,000 (s.380)"
    statute: str = ""            # optional, from the input only
    elements: list[str] = Field(default_factory=list)


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
    # The essential legal elements the prosecution/plaintiff must establish. The
    # judge charges the jury on these and jurors find each one proved or not — so
    # the verdict turns on the law, not on a holistic gut feeling.
    elements: list[str] = Field(default_factory=list)
    # Per-charge elements. With >1 charge each gets a separate verdict; `elements`
    # above stays as the primary charge's elements for single-charge compatibility.
    charges: list[Charge] = Field(default_factory=list)
    # The immutable fact ledger (see AgreedRecord) — the single source of truth.
    agreed_record: AgreedRecord = Field(default_factory=AgreedRecord)


class ElementFinding(BaseModel):
    """A juror's finding on ONE legal element for one accused.

    `proven` means the juror is satisfied that element is met to the applicable
    standard of proof. A single essential element left unproven means acquit.
    """

    element: str = ""
    proven: bool = False
    note: str = ""


class ChargeVote(BaseModel):
    """A juror's verdict on ONE charge (for one accused), with its own findings."""

    charge_label: str = ""
    verdict: str = "not guilty"
    vote: VoteChoice = "acquit"
    confidence: int = Field(default=5, ge=1, le=10)
    reasoning: str = ""
    element_findings: list[ElementFinding] = Field(default_factory=list)


class JurorPersona(BaseModel):
    name: str = "Juror"
    background: str = ""
    disposition: str = ""


class JuryPool(BaseModel):
    jurors: list[JurorPersona] = Field(default_factory=list)


class TrialCast(BaseModel):
    """Auto-cast personalities for the non-jury speaking roles."""

    crown: RolePersona = Field(default_factory=RolePersona)
    defense: RolePersona = Field(default_factory=RolePersona)
    judge: RolePersona = Field(default_factory=RolePersona)
    witnesses: list[RolePersona] = Field(default_factory=list)


class DefendantVote(BaseModel):
    """A juror's verdict for one co-accused (multi-defendant trials)."""

    defendant_name: str = ""
    verdict: str = "not guilty"
    vote: VoteChoice = "acquit"
    confidence: int = Field(default=5, ge=1, le=10)
    reasoning: str = ""
    # Per-element findings for THIS accused (drives the convict/acquit signal).
    element_findings: list[ElementFinding] = Field(default_factory=list)
    # Per-charge votes for THIS accused (multi-charge trials); empty otherwise.
    charge_votes: list[ChargeVote] = Field(default_factory=list)


class JurorVote(BaseModel):
    juror_name: str = "Juror"
    verdict: str = "not guilty"  # "guilty"/"not guilty" or "liable"/"not liable"
    vote: VoteChoice = "acquit"  # normalized convict/acquit signal used for tallying
    confidence: int = Field(default=5, ge=1, le=10)
    reasoning: str = ""
    # Per-element findings for the (single) accused; ignored in multi-defendant
    # trials, where each entry in `defendant_votes` carries its own findings.
    element_findings: list[ElementFinding] = Field(default_factory=list)
    # Per-charge votes for the (single) accused in multi-charge trials.
    charge_votes: list[ChargeVote] = Field(default_factory=list)
    # Populated only in multi-defendant trials (one entry per co-accused).
    defendant_votes: list[DefendantVote] = Field(default_factory=list)


class DeliberationRemark(BaseModel):
    """One juror's spoken turn in the jury room (not a vote — that comes after).

    `leaning` is a short, free-text indication of where the juror currently sits
    (e.g. "leaning acquit on Vance"); it colours the discussion but never feeds
    the tally — only the formal vote does.
    """

    juror_name: str = "Juror"
    statement: str = ""
    leaning: str = ""


class StrawMovement(BaseModel):
    """How the room moved between the pre-discussion straw poll and the final vote."""

    initial_tally: dict[str, int] = Field(default_factory=dict)
    final_tally: dict[str, int] = Field(default_factory=dict)
    flips: list[str] = Field(default_factory=list)  # e.g. "Juror X: convict → acquit"


class ChargeVerdict(BaseModel):
    """The jury's verdict on one charge (computed in code)."""

    charge_label: str
    tally: dict[str, int]
    outcome: str
    unanimous: bool
    hung: bool
    dissent_summary: str


class DefendantVerdict(BaseModel):
    """The jury's verdict for one co-accused."""

    defendant_name: str
    role: str = ""
    tally: dict[str, int]
    outcome: str
    unanimous: bool
    hung: bool
    dissent_summary: str
    # Per-charge breakdown for this accused (multi-charge trials).
    per_charge: list[ChargeVerdict] = Field(default_factory=list)


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
    # Per-charge breakdown for the (single) accused in multi-charge trials.
    per_charge: list[ChargeVerdict] = Field(default_factory=list)


class JudgeRuling(BaseModel):
    verdict_acknowledgement: str = ""
    reasoning: str = ""
    sentence_or_remedy: str = ""
    closing_remarks: str = ""
    # Structured sentencing detail — populated only on a conviction / liability,
    # grounded in the Agreed Record. Empty on acquittal / mistrial / dismissal.
    aggravating_factors: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    sentencing_range: str = ""           # e.g. "18-36 months custody (s.380 over $5k)"
    restitution: str = ""                # amount/terms, or victim-impact acknowledgement
    conditions: list[str] = Field(default_factory=list)  # probation terms / conditions


# ---------------------------------------------------------------------------
# Grounding / fact-check (anti-hallucination)
# ---------------------------------------------------------------------------
class GroundingFlag(BaseModel):
    """One claim in a statement that is not supported by the Agreed Record/record."""

    claim: str = ""
    issue: Literal[
        "unsupported", "fabricated", "misquoted_figure",
        "invented_authority", "contradicts_record",
    ] = "unsupported"
    severity: GroundingSeverity = "minor"
    explanation: str = ""
    record_basis: str = ""   # the nearest Record/transcript support, or "none"


class GroundingReport(BaseModel):
    """A fact-checker's verdict on a single statement."""

    speaker: str = ""
    phase: str = ""
    grounded: bool = True     # true when there are no material flags
    flags: list[GroundingFlag] = Field(default_factory=list)
    note: str = ""


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


class DirectedVerdictMotion(BaseModel):
    """Defence motion at the close of the prosecution's case (no-evidence motion)."""

    move: bool = False
    element_targeted: str = ""   # the essential element on which there is no evidence
    argument: str = ""


class DirectedVerdictRuling(BaseModel):
    """The judge's ruling on a directed-verdict motion."""

    granted: bool = False
    reasoning: str = ""
    # Names acquitted, if only some co-accused; empty = applies to all when granted.
    per_defendant: list[str] = Field(default_factory=list)


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

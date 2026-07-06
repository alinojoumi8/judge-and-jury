"""Trial state machine. Streams a sequence of TrialEvent objects."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .agents import (
    build_caster_agent,
    build_crown_agent,
    build_defense_agent,
    build_deliberation_agent,
    build_foreperson_agent,
    build_intake_agent,
    build_judge_agent,
    build_juror_agent,
    build_juror_pool_agent,
    build_ruling_agent,
    build_strategist_agent,
    build_verifier_agent,
    build_witness_agent,
)
from .llm_utils import run_structured
from .model_factory import build_model
from .schemas import (
    AgreedRecord,
    CaseInput,
    CaseStrategy,
    Charge,
    ChargeVerdict,
    Defendant,
    DefendantVerdict,
    DeliberationRemark,
    DirectedVerdictMotion,
    DirectedVerdictRuling,
    ExaminationQuestion,
    GroundingFlag,
    GroundingReport,
    JuryPool,
    JurorPersona,
    JurorVote,
    JudgeRuling,
    Objection,
    ObjectionRuling,
    RolePersona,
    Speech,
    StrawMovement,
    StructuredCase,
    TrialCast,
    TrialEvent,
    Verdict,
    WitnessAnswer,
)


def _strategy_text(s: CaseStrategy) -> str:
    """Render a counsel's persistent case theory for threading into their prompts."""
    pts = "; ".join(s.strongest_points) if s.strongest_points else "(none stated)"
    return (
        "YOUR CASE THEORY (hold to this consistently across opening, argument and "
        f"closing): {s.theory}\n"
        f"Your strongest points: {pts}\n"
        f"The opponent's single best point you must neutralise: {s.opponents_best_point}\n"
        f"Your planned answer to it: {s.rebuttal}"
    )


def _persona_text(p: RolePersona, role_label: str) -> str:
    """Render a role's personality for threading into their prompts ("" when empty)."""
    if not (p.name or p.background or p.style):
        return ""
    who = p.name or "you"
    bits = [f"YOU ARE {who} — {p.background}." if p.background else f"YOU ARE {who}."]
    if p.style:
        bits.append(f"Your manner as the {role_label}: {p.style}.")
    bits.append(
        "Stay in character consistently, in your own natural voice — do not announce "
        "or describe your personality, just embody it."
    )
    return " ".join(bits)


def _fallback_cast(witnesses: list) -> TrialCast:
    """Neutral default personalities, used only if casting generation fails."""
    return TrialCast(
        crown=RolePersona(
            name="Crown Counsel", background="a measured career prosecutor",
            style="methodical and understated; lets the facts do the work",
        ),
        defense=RolePersona(
            name="Defence Counsel", background="a seasoned defence advocate",
            style="warm and plain-spoken; presses reasonable doubt",
        ),
        judge=RolePersona(
            name="The Presiding Judge", background="a patient trial judge",
            style="even-handed and procedural",
        ),
        witnesses=[
            RolePersona(name=w.name, background=f"a {w.role} witness",
                        style="composed and straightforward")
            for w in witnesses
        ],
    )


def _record_block(rec: AgreedRecord) -> str:
    """Render the immutable Agreed Record + grounding rules for the case brief.

    Returns "" when the ledger is empty, so older cases add nothing to the brief.
    """
    if not any(
        (rec.parties, rec.figures, rec.dates, rec.admissible_facts, rec.authorities)
    ):
        return ""
    lines = ["=== THE AGREED RECORD (single source of truth; immutable) ==="]
    if rec.parties:
        lines.append("Parties: " + "; ".join(rec.parties))
    if rec.figures:
        lines.append("Key figures: " + "; ".join(rec.figures))
    if rec.dates:
        lines.append("Key dates: " + "; ".join(rec.dates))
    if rec.admissible_facts:
        lines.append("Admissible facts:")
        lines.extend(f"  {i}. {f}" for i, f in enumerate(rec.admissible_facts, 1))
    lines.append(
        "Cited authorities on record: "
        + ("; ".join(rec.authorities) if rec.authorities else "none")
    )
    lines.append(
        "GROUNDING RULES: Argue ONLY from this Record and on-record testimony. Do NOT "
        "invent facts, figures, dates, parties, or legal citations (statutes/case law) "
        "not listed above. Anything beyond the Record must be framed as argument or "
        'inference ("I submit…", "the inference is…"), never asserted as established fact.'
    )
    return "\n".join(lines)


def _exam_kinds(case: CaseInput) -> list[str]:
    """The witness-examination phases for this case (pure, for testing)."""
    kinds = ["DIRECT", "CROSS"]
    if case.redirect and case.qa_redirect:
        kinds.append("REDIRECT")
    return kinds


def _directed_acquittal(defendants: list, case_type: str) -> Verdict:
    """Build an acquittal Verdict for a granted directed-verdict motion (no jury vote)."""
    acquit_word = "Not Liable" if case_type == "civil" else "Not Guilty"
    convict_word = "Liable" if case_type == "civil" else "Guilty"
    tally = {convict_word: 0, acquit_word: 0}
    dissent = "Directed verdict — the case was withdrawn from the jury."
    if len(defendants) > 1:
        per = [
            DefendantVerdict(
                defendant_name=d.name, role=getattr(d, "role", ""),
                tally=dict(tally), outcome=acquit_word, unanimous=True, hung=False,
                dissent_summary=dissent,
            )
            for d in defendants
        ]
        head = per[0]
        return Verdict(
            tally=head.tally, outcome=head.outcome, unanimous=True, hung=False,
            dissent_summary=dissent, per_defendant=per,
        )
    return Verdict(
        tally=tally, outcome=acquit_word, unanimous=True, hung=False,
        dissent_summary=dissent,
    )


def _charge_directives(case: CaseInput, defendants: list) -> list[str]:
    """The judge's extra charge instructions for this case (W.(D.), circumstantial…).

    Pure so it can be unit-tested. Criminal cases get the full set; civil gets only
    the inference instruction; party-liability is added only with co-accused.
    """
    out: list[str] = []
    if case.case_type == "criminal":
        out.append(
            "Circumstantial evidence: the jury may convict on circumstantial evidence "
            "ONLY if guilt is the only reasonable inference; if any other reasonable "
            "inference is consistent with innocence, they must acquit."
        )
        out.append(
            "Credibility (the W.(D.) approach): if they believe the accused's evidence "
            "they must acquit; if they do not believe it but it leaves a reasonable "
            "doubt, acquit; even if it leaves no doubt, convict only if the evidence "
            "they DO accept proves guilt beyond a reasonable doubt."
        )
        if len(defendants) > 1:
            out.append(
                "Party liability (Criminal Code s.21): a person is a party whether they "
                "personally commit the offence, aid, or abet it — address each accused's "
                "role separately."
            )
    out.append(
        "Permissible inferences: the jury may draw reasonable inferences from proven "
        "facts, but must not speculate or fill gaps with guesswork."
    )
    return out


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------
def _normalize_vote(verdict_text: str, model_vote: str) -> str:
    """Return a reliable 'convict' / 'acquit' signal for tallying.

    Prefers the model's explicit enum, but if the free-text verdict UNAMBIGUOUSLY
    disagrees, trust the text. Deliberately less brittle than a bare "contains
    'not'" check (which mis-reads e.g. "cannot be doubted, guilty" as acquittal).
    """
    t = (verdict_text or "").strip().lower()
    acquit_text = (
        "not guilty" in t or "not liable" in t or "acquit" in t or "innocent" in t
    )
    convict_text = (not acquit_text) and (
        "guilty" in t or "liable" in t or "convict" in t
    )
    if acquit_text:
        return "acquit"
    if convict_text:
        return "convict"
    return "convict" if model_vote == "convict" else "acquit"


def _derive_vote(findings: list, model_vote: str, verdict_text: str) -> str:
    """Turn per-element findings into a 'convict'/'acquit' signal for tallying.

    When the juror worked through the legal elements, those are authoritative: the
    accused is convicted ONLY if every element is proven — a single unproven element
    means acquit. With no element findings, fall back to the free-text/enum reading.
    """
    if findings:
        return "convict" if all(getattr(f, "proven", False) for f in findings) else "acquit"
    return _normalize_vote(verdict_text, model_vote)


def _settled_with_conviction(votes: list[JurorVote], verdict, min_conf: int = 6) -> bool:
    """True only if the side that carried the verdict holds it with real conviction.

    An extra gate on early-exit: a unanimous/majority outcome held with LOW average
    confidence should get another (bounded) deliberation round rather than ending
    immediately. A genuinely confident room exits. `verdict` is accepted for API
    symmetry; the carrying side is read from the ballots so it works pre-tally too.
    """
    if not votes:
        return True
    convict_n = sum(1 for v in votes if v.vote == "convict")
    carrying = "convict" if convict_n * 2 > len(votes) else "acquit"
    side = [v for v in votes if v.vote == carrying]
    if not side:
        return True
    mean_conf = sum(v.confidence for v in side) / len(side)
    return mean_conf >= min_conf


def _straw_movement(
    initial: list[JurorVote], final: list[JurorVote],
    defendants: list, charges: list, case_type: str,
) -> StrawMovement:
    """Diff the pre-discussion straw poll against the final vote (which jurors moved).

    Uses _tally (not the flat _tally_votes) so multi-defendant / multi-charge cases
    show the first-defendant summary tally, consistent with the straw-poll event itself.
    """
    init_by = {v.juror_name: v.vote for v in initial}
    flips: list[str] = []
    for v in final:
        before = init_by.get(v.juror_name)
        if before and before != v.vote:
            flips.append(f"{v.juror_name}: {before} → {v.vote}")
    return StrawMovement(
        initial_tally=_tally(initial, defendants, charges, case_type).tally if initial else {},
        final_tally=_tally(final, defendants, charges, case_type).tally if final else {},
        flips=flips,
    )


def _merge_reports(reports: list, speaker: str = "", phase: str = "") -> GroundingReport:
    """Union the flags from one or more fact-checkers; dedup by claim, keep max severity."""
    order = {"minor": 0, "moderate": 1, "severe": 2}
    best: dict[str, GroundingFlag] = {}
    for r in reports:
        if r is None:
            continue
        for f in r.flags:
            key = f.claim.strip().lower()
            prev = best.get(key)
            if prev is None or order.get(f.severity, 0) > order.get(prev.severity, 0):
                best[key] = f
    flags = list(best.values())
    return GroundingReport(speaker=speaker, phase=phase, grounded=not flags, flags=flags)


def _tally_votes(votes: list[JurorVote], case_type: str) -> Verdict:
    convict_word = "Liable" if case_type == "civil" else "Guilty"
    acquit_word = "Not Liable" if case_type == "civil" else "Not Guilty"

    convict = sum(1 for v in votes if v.vote == "convict")
    acquit = len(votes) - convict
    tally = {convict_word: convict, acquit_word: acquit}

    unanimous = convict == 0 or acquit == 0

    if case_type == "civil":
        # Civil matters are decided on the balance of probabilities: a majority
        # carries, and only an even split leaves the jury hung.
        hung = convict == acquit
        if hung:
            outcome = "Hung jury (no majority)"
        elif convict > acquit:
            outcome = convict_word
        else:
            outcome = acquit_word
    else:
        # A criminal verdict must be UNANIMOUS. Any division — even a lopsided
        # majority — is a hung jury, not a verdict.
        hung = not unanimous
        if hung:
            outcome = "Hung jury (no unanimous verdict)"
        elif convict > 0:
            outcome = convict_word
        else:
            outcome = acquit_word

    if hung and case_type == "civil":
        dissent_summary = "The jury was evenly split and could not reach a majority."
    elif hung:
        tally_line = f"{convict} {convict_word} – {acquit} {acquit_word}"
        dissent_summary = (
            "A criminal verdict must be unanimous; the jury was divided "
            f"({tally_line}) and could not agree."
        )
    elif unanimous:
        dissent_summary = "The verdict was unanimous."
    else:
        # Civil majority verdict — name the dissenting minority.
        losing = acquit_word if outcome == convict_word else convict_word
        dissenters = [
            v for v in votes
            if (v.vote == "acquit" and outcome == convict_word)
            or (v.vote == "convict" and outcome == acquit_word)
        ]
        names = ", ".join(v.juror_name for v in dissenters)
        dissent_summary = f"Majority verdict. Dissenting ({losing}): {names}."

    return Verdict(
        tally=tally,
        outcome=outcome,
        unanimous=unanimous,
        hung=hung,
        dissent_summary=dissent_summary,
    )


def _project_defendant_ballots(votes: list[JurorVote], d: Defendant) -> list[JurorVote]:
    """Each juror's ballot FOR ONE accused — their DefendantVote, or the top-level.

    Carries element_findings and per-charge votes through so a downstream per-charge
    tally still works. Missing/misspelled entries fall back to the top-level vote.
    """
    out: list[JurorVote] = []
    for v in votes:
        m = next(
            (x for x in v.defendant_votes
             if x.defendant_name.strip().lower() == d.name.strip().lower()),
            None,
        )
        src = m if m is not None else v
        out.append(
            JurorVote(
                juror_name=v.juror_name,
                verdict=src.verdict,
                vote=src.vote,
                confidence=src.confidence,
                reasoning=src.reasoning,
                element_findings=list(getattr(src, "element_findings", [])),
                charge_votes=list(getattr(src, "charge_votes", [])),
            )
        )
    return out


def _tally_votes_multi(
    votes: list[JurorVote], defendants: list[Defendant], case_type: str
) -> Verdict:
    """Tally a separate verdict for each co-accused from one set of juror ballots."""
    per: list[DefendantVerdict] = []
    for d in defendants:
        base = _tally_votes(_project_defendant_ballots(votes, d), case_type)
        per.append(
            DefendantVerdict(
                defendant_name=d.name,
                role=d.role,
                **base.model_dump(exclude={"per_defendant", "per_charge"}),
            )
        )
    head = per[0]
    return Verdict(
        tally=head.tally,
        outcome=head.outcome,
        unanimous=head.unanimous,
        hung=head.hung,
        dissent_summary=head.dissent_summary,
        per_defendant=per,
    )


def _tally_votes_by_charge(
    votes: list[JurorVote], charges: list[Charge], case_type: str
) -> list[ChargeVerdict]:
    """A separate verdict for EACH charge, projecting each juror's per-charge vote.

    Mirrors the per-defendant projection: match on charge label, fall back to the
    juror's top-level vote when an entry is missing.
    """
    out: list[ChargeVerdict] = []
    for c in charges:
        projected: list[JurorVote] = []
        for v in votes:
            m = next(
                (x for x in v.charge_votes
                 if x.charge_label.strip().lower() == c.label.strip().lower()),
                None,
            )
            src = m if m is not None else v
            projected.append(
                JurorVote(
                    juror_name=v.juror_name, verdict=src.verdict, vote=src.vote,
                    confidence=src.confidence, reasoning=src.reasoning,
                )
            )
        base = _tally_votes(projected, case_type)
        out.append(
            ChargeVerdict(
                charge_label=c.label, tally=base.tally, outcome=base.outcome,
                unanimous=base.unanimous, hung=base.hung, dissent_summary=base.dissent_summary,
            )
        )
    return out


def _tally(
    votes: list[JurorVote], defendants: list[Defendant], charges: list[Charge], case_type: str
) -> Verdict:
    """Compose the two verdict axes: per-defendant (outer) and per-charge (inner)."""
    multi_c = len(charges) > 1
    if len(defendants) > 1:
        v = _tally_votes_multi(votes, defendants, case_type)
        if multi_c:
            for dv, d in zip(v.per_defendant, defendants):
                dv.per_charge = _tally_votes_by_charge(
                    _project_defendant_ballots(votes, d), charges, case_type
                )
        return v
    base = _tally_votes(votes, case_type)
    if multi_c:
        base.per_charge = _tally_votes_by_charge(votes, charges, case_type)
    return base


def _tally_signature(votes: list[JurorVote]) -> tuple:
    """A hashable fingerprint of the jury's current position (per defendant/charge).

    Used to stop deliberation early when a divided jury stops moving. Descends into
    per-charge votes when present so a charge-by-charge shift still registers.
    """
    def charge_sig(obj):
        cvs = getattr(obj, "charge_votes", [])
        if cvs:
            return tuple(sorted((cv.charge_label.strip().lower(), cv.vote) for cv in cvs))
        return getattr(obj, "vote", "acquit")

    sig = []
    for v in votes:
        if v.defendant_votes:
            inner = tuple(
                sorted((dv.defendant_name.strip().lower(), charge_sig(dv)) for dv in v.defendant_votes)
            )
            sig.append((v.juror_name, inner))
        else:
            sig.append((v.juror_name, charge_sig(v)))
    return tuple(sorted(sig, key=lambda x: x[0]))


def _all_settled(verdict: Verdict) -> bool:
    """True if there is nothing left to deliberate (every verdict is unanimous)."""
    if verdict.per_defendant:
        return all(
            d.unanimous and all(c.unanimous for c in d.per_charge)
            for d in verdict.per_defendant
        )
    if verdict.per_charge:
        return all(c.unanimous for c in verdict.per_charge)
    return verdict.unanimous


def _defendant_roster(case: CaseInput, sc: StructuredCase) -> list[Defendant]:
    """The co-accused to judge. Falls back to one synthetic defendant (today's flow)."""
    if case.defendants:
        return case.defendants[:4]  # hard cap N
    caption = sc.case_caption or ""
    name = "the accused"
    for sep in (" v. ", " v ", " vs. ", " vs "):
        if sep in caption:
            name = caption.split(sep)[-1].strip() or name
            break
    return [Defendant(name=name, role="", account=case.your_side)]


def _fallback_personas(n: int) -> list[JurorPersona]:
    base = [
        ("Alex Morgan", "retired schoolteacher", "values clear evidence"),
        ("Priya Patel", "software engineer", "logical and detail-oriented"),
        ("Sam Okonkwo", "small-business owner", "practical, trusts common sense"),
        ("Maria Garcia", "nurse", "empathetic, weighs harm to people"),
        ("Tom Becker", "construction foreman", "skeptical of fancy arguments"),
        ("Lena Novak", "journalist", "probes for inconsistencies"),
        ("David Chen", "accountant", "wants the numbers to add up"),
        ("Grace Bauer", "social worker", "attentive to context and motive"),
        ("Omar Haddad", "taxi driver", "street-smart, distrusts spin"),
        ("Nora Fitzgerald", "librarian", "careful and methodical"),
        ("Yuki Tanaka", "chef", "decisive once convinced"),
        ("Paul Andersson", "electrician", "no-nonsense, wants hard proof"),
    ]
    return [JurorPersona(name=n_, background=b, disposition=d) for n_, b, d in base[:n]]


# ---------------------------------------------------------------------------
# Main state machine
# ---------------------------------------------------------------------------
async def run_trial(case: CaseInput) -> AsyncIterator[TrialEvent]:
    try:
        async for ev in _run_trial_inner(case):
            yield ev
    except Exception as exc:  # surface any failure as an event instead of a 500
        yield TrialEvent(
            phase="error",
            speaker="System",
            kind="error",
            content=f"{type(exc).__name__}: {exc}",
        )
    yield TrialEvent(phase="done", speaker="System", kind="done")


async def _run_trial_inner(case: CaseInput) -> AsyncIterator[TrialEvent]:
    model = build_model(case.model)
    with_witnesses = bool(case.witnesses)

    intake = build_intake_agent(model)
    crown = build_crown_agent(model, with_witnesses=with_witnesses)
    defense = build_defense_agent(model, with_witnesses=with_witnesses)
    judge = build_judge_agent(model, with_witnesses=with_witnesses)
    ruling_agent = build_ruling_agent(model)
    pool_agent = build_juror_pool_agent(model)
    juror_agent = build_juror_agent(model)
    verifier = build_verifier_agent(model)

    transcript: list[str] = []

    def log(speaker: str, text: str) -> None:
        transcript.append(f"{speaker}: {text}")

    def transcript_text() -> str:
        return "\n\n".join(transcript) if transcript else "(no statements yet)"

    convict_word = "Liable" if case.case_type == "civil" else "Guilty"
    acquit_word = "Not Liable" if case.case_type == "civil" else "Not Guilty"
    standard = (
        "balance of probabilities"
        if case.case_type == "civil"
        else "beyond a reasonable doubt"
    )
    # "Crown" is criminal-prosecution terminology; in a civil matter the opposing
    # side is the plaintiff (e.g. a regulator such as a securities commission).
    prosecutor = "Plaintiff" if case.case_type == "civil" else "Crown"

    async def speak(
        agent, prompt: str, *, phase: str, speaker: str, self_check: bool = False
    ) -> TrialEvent:
        """Run a free-form role and return its complete statement as a message event.

        With `self_check=True` and `case.self_ground`, the role drafts, then revises
        its statement against the Agreed Record (recasting unsupported assertions as
        explicit argument) before it is logged — a cheap anti-hallucination pass.
        """
        try:
            text = (
                await run_structured(agent, prompt, Speech, require_english=True)
            ).statement.strip()
            if self_check and case.self_ground:
                revise = (
                    f"{prompt}\n\nYOUR DRAFT:\n{text}\n\n{record_block}\n\nSELF-CHECK: revise "
                    "your statement so EVERY factual assertion, figure, date and citation is "
                    "supported by the Agreed Record or on-record testimony. Recast anything "
                    "unsupported as explicit argument (\"I submit…\") or delete it. Return the "
                    "revised statement only."
                )
                text = (
                    await run_structured(agent, revise, Speech, require_english=True)
                ).statement.strip()
        except Exception as exc:  # degrade gracefully; keep the trial going
            text = f"(The {speaker} was unable to make a statement: {exc})"
        log(speaker, text)
        return TrialEvent(phase=phase, speaker=speaker, kind="message", content=text)

    # --- 1. Intake ------------------------------------------------------
    yield TrialEvent(phase="Intake", speaker="System", kind="phase", content="Case Intake")
    intake_prompt = (
        f"Case type: {case.case_type}\n"
        f"Jurisdiction: {case.jurisdiction}\n"
        f"Charge/claim: {case.charge_or_claim}\n"
        f"Client's account:\n{case.your_side}"
    )
    if case.defendants:
        roster = "\n".join(
            f"- {d.name} ({d.role or 'role unstated'}): {d.account}" for d in case.defendants
        )
        intake_prompt += f"\n\nCo-accused (each judged separately):\n{roster}"
    sc: StructuredCase = await run_structured(
        intake, intake_prompt, StructuredCase, require_english=True
    )
    log("Court Clerk", f"Case filed: {sc.case_caption}. {sc.summary}")
    yield TrialEvent(
        phase="Intake",
        speaker="Law Firm Clerk",
        kind="structured",
        content=sc.summary,
        data=sc.model_dump(),
    )

    # The immutable Agreed Record — the single source of truth, threaded into every
    # later prompt via `case_brief` so the whole trial is grounded.
    record_block = _record_block(sc.agreed_record)
    if record_block:
        yield TrialEvent(
            phase="Intake",
            speaker="Court Clerk",
            kind="structured",
            content="The Agreed Record has been settled.",
            data={"agreed_record": sc.agreed_record.model_dump(), "_record": True},
        )

    async def ground_event(statement, *, speaker, phase, gphase, checkers=1):
        """Fact-check a statement against the Agreed Record; return a flag event or None.

        No-op unless `grounding_check` is on and this phase is in `grounding_phases`.
        Never alters the trial — only surfaces ungrounded claims.
        """
        if not case.grounding_check or gphase not in case.grounding_phases:
            return None
        prompt = (
            f"{record_block or '(no agreed record was settled)'}\n\nTRANSCRIPT SO FAR:\n"
            f"{transcript_text()}\n\nSTATEMENT BY {speaker} (phase: {phase}):\n{statement}\n\n"
            "Return ONLY a GroundingReport JSON."
        )
        try:
            raw = await asyncio.gather(*[
                run_structured(verifier, prompt, GroundingReport, require_english=True)
                for _ in range(max(1, checkers))
            ], return_exceptions=True)
            reports = [r for r in raw if not isinstance(r, BaseException)]
            if not reports:
                return None
        except Exception:
            return None
        report = _merge_reports(reports, speaker, phase)
        if not report.flags:
            return None
        return TrialEvent(
            phase=phase, speaker="Fact-Check", kind="structured",
            content=f"{len(report.flags)} grounding flag(s) on {speaker}'s statement.",
            data={**report.model_dump(), "_grounding": True},
        )

    defendants = _defendant_roster(case, sc)
    multi = len(defendants) > 1

    # Charges with per-charge elements. A single charge (or none) collapses to one
    # Charge carrying the flat `elements`, so single-charge cases behave as before.
    charges = list(sc.charges)
    if len(charges) <= 1:
        label = (
            charges[0].label if (charges and charges[0].label)
            else (sc.charges_or_claims[0] if sc.charges_or_claims else "the charge")
        )
        single_elements = charges[0].elements if (charges and charges[0].elements) else sc.elements
        charges = [Charge(label=label, elements=single_elements)]
    multi_c = len(charges) > 1
    who_proves = "plaintiff" if case.case_type == "civil" else "Crown"

    if multi_c:
        parts = [
            f"\nThe {who_proves} must prove EACH charge separately to the standard of "
            f"{standard}; for each charge, failure on ANY of its elements means acquittal "
            "ON THAT CHARGE:"
        ]
        for ci, c in enumerate(charges, 1):
            parts.append(f"  Charge {ci} — {c.label}:")
            parts.extend(f"    {ei}. {e}" for ei, e in enumerate(c.elements, 1))
        elements_block = "\n".join(parts)
    else:
        single_elements = charges[0].elements
        elements_block = (
            f"\nELEMENTS the {who_proves} must prove (each to the standard of {standard}; "
            "failure on ANY one means the accused must be acquitted):\n"
            + "\n".join(f"  {i}. {e}" for i, e in enumerate(single_elements, 1))
            if single_elements
            else ""
        )
    case_brief = (
        (f"{record_block}\n\n" if record_block else "")
        + f"CASE: {sc.case_caption}\n"
        f"Type: {case.case_type} | Jurisdiction: {case.jurisdiction} | "
        f"Standard of proof: {standard}\n"
        f"Charges/Claims: {'; '.join(sc.charges_or_claims)}\n"
        f"Summary: {sc.summary}\n"
        f"Key facts: {'; '.join(sc.key_facts)}\n"
        f"Prosecution theory: {sc.prosecution_theory}\n"
        f"Defense theory: {sc.defense_theory}"
        f"{elements_block}"
    )
    if multi:
        roster = "\n".join(
            f"- {d.name} ({d.role or 'role unstated'}): {d.account}" for d in defendants
        )
        case_brief += (
            "\n\nCO-ACCUSED (each must receive a SEPARATE verdict, judged on their "
            f"own role and conduct):\n{roster}"
        )

    # --- 1b. Counsel's persistent trial strategy ------------------------
    # Each side forms a theory ONCE and carries it through opening, argument and
    # closing — so their position stays coherent and they can steelman + rebut the
    # opponent's strongest point instead of re-deriving the case every turn.
    strategist = build_strategist_agent(model)

    async def _make_strategy(side_label: str) -> CaseStrategy:
        try:
            return await run_structured(
                strategist,
                f"You act for the {side_label}. Prepare your case theory for THIS case.\n\n"
                f"{case_brief}",
                CaseStrategy,
                require_english=True,
            )
        except Exception:
            return CaseStrategy()

    crown_strategy = await _make_strategy(f"prosecution ({prosecutor})")
    defense_strategy = await _make_strategy("defense")
    crown_strategy_text = _strategy_text(crown_strategy)
    defense_strategy_text = _strategy_text(defense_strategy)

    yield TrialEvent(
        phase="Strategy", speaker="System", kind="phase", content="Counsel's Trial Strategy"
    )
    yield TrialEvent(
        phase="Strategy", speaker=prosecutor, kind="structured",
        content=crown_strategy.theory,
        data={**crown_strategy.model_dump(), "_strategy": True, "side": prosecutor},
    )
    yield TrialEvent(
        phase="Strategy", speaker="Defense", kind="structured",
        content=defense_strategy.theory,
        data={**defense_strategy.model_dump(), "_strategy": True, "side": "Defense"},
    )

    # --- 1c. Casting: a distinct personality for each non-jury speaking role ----
    # Pinned personas (from CaseInput) override auto-casting; when `personas` is on,
    # the casting director fills any unpinned role + each witness's demeanour. Each
    # persona is threaded into that role's prompts so they speak in a consistent
    # voice. With no personas and no pins, roles keep their generic voices.
    crown_p = case.crown_persona or RolePersona()
    defense_p = case.defense_persona or RolePersona()
    judge_p = case.judge_persona or RolePersona()
    cast_witnesses: list[RolePersona] = []
    witness_demeanour: dict[str, str] = {}  # lowercased witness name -> demeanour
    if case.personas:
        caster = build_caster_agent(model)
        _wnames = ", ".join(w.name for w in case.witnesses[: case.max_witnesses]) or "(none)"
        try:
            cast = await run_structured(
                caster,
                f"{case_brief}\n\nWitnesses to cast (echo each name EXACTLY): {_wnames}\n\n"
                "Cast the prosecutor, the defence counsel, the judge, and a courtroom "
                "demeanour for each witness.",
                TrialCast, require_english=True,
            )
        except Exception:
            cast = _fallback_cast(case.witnesses[: case.max_witnesses])
        crown_p = case.crown_persona or cast.crown        # a pinned persona always wins
        defense_p = case.defense_persona or cast.defense
        judge_p = case.judge_persona or cast.judge
        cast_witnesses = cast.witnesses
        for cw in cast.witnesses:
            if cw.name:
                witness_demeanour[cw.name.strip().lower()] = cw.style
    crown_persona_text = _persona_text(crown_p, prosecutor)
    defense_persona_text = _persona_text(defense_p, "defence counsel")
    judge_persona_text = _persona_text(judge_p, "judge")
    if case.personas or case.crown_persona or case.defense_persona or case.judge_persona:
        final_cast = TrialCast(
            crown=crown_p, defense=defense_p, judge=judge_p, witnesses=cast_witnesses
        )
        yield TrialEvent(phase="Strategy", speaker="System", kind="phase", content="The Cast")
        yield TrialEvent(
            phase="Strategy", speaker="Court Clerk", kind="structured",
            content="Counsel and the bench have taken their places.",
            data={"_cast": True, **final_cast.model_dump()},
        )

    def _demeanour(w) -> str:
        return w.demeanour or witness_demeanour.get(w.name.strip().lower(), "")

    # --- 2. Jury selection ----------------------------------------------
    yield TrialEvent(
        phase="Jury Selection", speaker="System", kind="phase", content="Jury Selection"
    )
    try:
        pool: JuryPool = await run_structured(
            pool_agent,
            f"Select exactly {case.jury_size} jurors (N={case.jury_size}) for this "
            f"{case.case_type} trial.",
            JuryPool,
            require_english=True,
        )
        personas = pool.jurors[: case.jury_size]
    except Exception:
        personas = []
    if len(personas) < case.jury_size:
        personas = _fallback_personas(case.jury_size)

    yield TrialEvent(
        phase="Jury Selection",
        speaker="Court Clerk",
        kind="structured",
        content=f"A jury of {len(personas)} has been empanelled.",
        data={"jurors": [p.model_dump() for p in personas]},
    )

    # --- 3. Judge opening ------------------------------------------------
    yield TrialEvent(phase="Opening", speaker="System", kind="phase", content="The Trial Begins")
    _directives = _charge_directives(case, defendants)
    charge_block = (
        "Include these directives in your charge to the jury:\n- "
        + "\n- ".join(_directives)
        + "\n\n"
        if _directives
        else ""
    )
    yield await speak(
        judge,
        f"{judge_persona_text}\n\nOpen this trial and instruct the jury.\n\n{charge_block}{case_brief}",
        phase="Opening",
        speaker="Judge",
    )

    # --- 4. Opening statements ------------------------------------------
    yield await speak(
        crown,
        f"Deliver your OPENING STATEMENT.\n\n{crown_persona_text}\n\n{crown_strategy_text}\n\n{case_brief}",
        phase="Opening",
        speaker=prosecutor,
    )
    yield await speak(
        defense,
        f"Deliver your OPENING STATEMENT.\n\n{defense_persona_text}\n\n{defense_strategy_text}\n\n{case_brief}\n\n"
        f"Transcript so far:\n{transcript_text()}",
        phase="Opening",
        speaker="Defense",
    )

    # --- 4b. Witness testimony & cross-examination (optional) -----------
    if with_witnesses:
        witness_agent = build_witness_agent(model)

        async def examine(examiner, examiner_name, objector, objector_name, w, witness_brief, kind, testimony, n=None):
            n_q = n or case.qa_exchanges
            yield TrialEvent(
                phase="Evidence", speaker="Court Clerk", kind="message",
                content=f"{kind.title()} examination of {w.name} by {examiner_name}.",
            )
            for i in range(1, n_q + 1):
                # The full running record of THIS witness's testimony (direct + cross
                # so far). Both the examiner and the witness see it, so cross-
                # examination can probe — and the witness can stay consistent.
                record = "\n".join(testimony) or "(nothing yet)"
                try:
                    q = (await run_structured(
                        examiner,
                        f"{kind} EXAMINATION of witness {w.name} ({w.role}). Ask question "
                        f"{i} of {n_q}. "
                        + ("Build your case." if kind == "DIRECT"
                           else "Re-examination — confine yourself to matters raised on "
                                "cross-examination; rehabilitate the witness on points the cross "
                                "damaged, and do NOT open new topics." if kind == "REDIRECT"
                           else "Test or impeach the witness — if the testimony so far contains a "
                                "weakness, inconsistency, or useful concession, put a pointed "
                                "question on it.")
                        + '\nReply with ONLY {"question": "your question"}.\n\n'
                        + f"{case_brief}\n\nWitness {w.name}'s testimony so far:\n{record}",
                        ExaminationQuestion, require_english=True,
                    )).question.strip()
                except Exception as exc:
                    q = f"(No question put: {exc})"
                testimony.append(f"Q — {examiner_name}: {q}")
                yield TrialEvent(
                    phase="Evidence", speaker=examiner_name, kind="message",
                    content=q, data={"witness": w.name, "exam": kind},
                )

                # Either side may object to a question (direct, cross, or re-direct).
                if kind in ("CROSS", "DIRECT", "REDIRECT"):
                    try:
                        obj = await run_structured(
                            objector,
                            f'You are opposing counsel. Opposing counsel asked witness '
                            f'{w.name}: "{q}". If the question is objectionable (leading / '
                            "hearsay / speculation / relevance) raise ONE short objection, "
                            "otherwise set object=false.\nReply with ONLY "
                            '{"object": true/false, "ground": "...", "text": "..."}.\n\n'
                            f"{witness_brief}",
                            Objection, require_english=True,
                        )
                    except Exception:
                        obj = Objection(object=False)
                    if obj.object:
                        yield TrialEvent(
                            phase="Evidence", speaker=objector_name, kind="message",
                            content=f"Objection — {obj.ground}. {obj.text}".strip(),
                            data={"objection": True},
                        )
                        try:
                            rule = await run_structured(
                                judge,
                                f'Counsel objected ({obj.ground}): "{obj.text}" to the question '
                                f'"{q}". Rule "sustained" or "overruled" in one short sentence.\n'
                                'Reply with ONLY {"ruling": "sustained" or "overruled", "text": "..."}.',
                                ObjectionRuling, require_english=True,
                            )
                        except Exception:
                            rule = ObjectionRuling(ruling="overruled", text="")
                        yield TrialEvent(
                            phase="Evidence", speaker="Judge", kind="structured",
                            content=f"{rule.ruling.title()}. {rule.text}".strip(),
                            data={"objection_ruling": rule.ruling, "text": rule.text},
                        )
                        if rule.ruling.lower().startswith("sustain"):
                            testimony.append("(Objection sustained; the question is withdrawn.)")
                            continue

                try:
                    a = (await run_structured(
                        witness_agent,
                        f"You are {w.name} on the stand under {kind} examination. Answer "
                        f'truthfully, IN CHARACTER, only from what you know, and CONSISTENTLY '
                        f'with your testimony so far. Question: "{q}".'
                        f"\n\n{witness_brief}\n\nYour testimony so far:\n{record}",
                        WitnessAnswer, require_english=True,
                    )).statement.strip()
                except Exception as exc:
                    a = f"(The witness could not answer: {exc})"
                testimony.append(f"A — {w.name}: {a}")
                log(f"Witness {w.name}", a)
                yield TrialEvent(
                    phase="Evidence", speaker=f"Witness — {w.name}", kind="message",
                    content=a, data={"witness": w.name},
                )
                wfc = await ground_event(
                    a, speaker=f"Witness — {w.name}", phase="Evidence", gphase="witness"
                )
                if wfc:
                    yield wfc

        yield TrialEvent(
            phase="Evidence", speaker="System", kind="phase",
            content="Witness Testimony & Cross-Examination",
        )
        for w in case.witnesses[: case.max_witnesses]:
            if w.called_by == "prosecution":
                caller, caller_name, opp, opp_name = crown, prosecutor, defense, "Defense"
            else:
                caller, caller_name, opp, opp_name = defense, "Defense", crown, prosecutor
            wd = _demeanour(w)
            witness_brief = (
                f"You are {w.name}, a {w.role} witness called by the {w.called_by}.\n"
                + (f"Your demeanour on the stand: {wd}.\n" if wd else "")
                + f"What you know:\n{w.what_they_know}\n\n{case_brief}"
            )
            # One running record per witness, carried from direct INTO cross so both
            # the cross-examiner and the witness remember the direct testimony.
            testimony: list[str] = []
            yield TrialEvent(
                phase="Evidence", speaker="Court Clerk", kind="message",
                content=f"{w.name} is called to the stand ({w.role}, for the {w.called_by}).",
            )
            async for ev in examine(caller, caller_name, opp, opp_name, w, witness_brief, "DIRECT", testimony):
                yield ev
            async for ev in examine(opp, opp_name, caller, caller_name, w, witness_brief, "CROSS", testimony):
                yield ev
            # Re-examination by the calling side, confined to matters raised on cross.
            if case.redirect and case.qa_redirect:
                async for ev in examine(
                    caller, caller_name, opp, opp_name, w, witness_brief, "REDIRECT",
                    testimony, n=case.qa_redirect,
                ):
                    yield ev

    # --- 4c. Directed-verdict motion (close of the prosecution's case) --
    if with_witnesses and case.allow_directed_verdict:
        yield TrialEvent(
            phase="Motion", speaker="System", kind="phase",
            content="Motion for a Directed Verdict",
        )
        try:
            motion = await run_structured(
                defense,
                "The prosecution has closed its case. If — taking the evidence at its "
                "highest — there is NO evidence on an essential element such that no "
                "reasonable jury could convict, move for a directed verdict naming that "
                'element; otherwise set move=false.\nReply with ONLY {"move": true/false, '
                '"element_targeted": "...", "argument": "..."}.\n\n'
                f"{case_brief}\n\nTranscript so far:\n{transcript_text()}",
                DirectedVerdictMotion, require_english=True,
            )
        except Exception:
            motion = DirectedVerdictMotion()
        if motion.move:
            yield TrialEvent(
                phase="Motion", speaker="Defense", kind="message",
                content=(
                    f"Motion for a directed verdict — no evidence on: "
                    f"{motion.element_targeted}. {motion.argument}"
                ).strip(),
            )
            log("Defense", f"Directed-verdict motion ({motion.element_targeted}).")
            try:
                dvr = await run_structured(
                    judge,
                    "Rule on the directed-verdict motion. Grant ONLY if there is truly no "
                    "evidence on the named essential element on which a reasonable jury, "
                    "properly instructed, could convict; otherwise dismiss and let the trial "
                    'continue.\nReply with ONLY {"granted": true/false, "reasoning": "...", '
                    '"per_defendant": ["names acquitted, if only some — else empty"]}.\n\n'
                    f"{case_brief}\n\nTranscript:\n{transcript_text()}\n\n"
                    f"Motion: {motion.element_targeted} — {motion.argument}",
                    DirectedVerdictRuling, require_english=True,
                )
            except Exception:
                dvr = DirectedVerdictRuling(granted=False)
            yield TrialEvent(
                phase="Motion", speaker="Judge", kind="structured",
                content="Directed verdict GRANTED." if dvr.granted else "Motion dismissed.",
                data={**dvr.model_dump(), "_directed_verdict": True},
            )
            log("Judge", ("Directed verdict granted. " if dvr.granted else "Motion dismissed. ") + dvr.reasoning)
            if dvr.granted:
                named = (
                    [d for d in defendants if d.name in dvr.per_defendant]
                    if dvr.per_defendant else list(defendants)
                )
                partial = bool(named) and len(named) < len(defendants)
                if partial:
                    # Acquit the named accused; the trial continues for the rest.
                    acq = _directed_acquittal(named, case.case_type)
                    yield TrialEvent(
                        phase="Verdict", speaker="System", kind="phase",
                        content="Directed Acquittal",
                    )
                    for dv in acq.per_defendant or [
                        DefendantVerdict(
                            defendant_name=named[0].name, tally=acq.tally, outcome=acq.outcome,
                            unanimous=True, hung=False, dissent_summary=acq.dissent_summary,
                        )
                    ]:
                        yield TrialEvent(
                            phase="Verdict", speaker=f"Verdict — {dv.defendant_name}",
                            kind="structured", content=f"{dv.defendant_name}: {dv.outcome} (directed)",
                            data={**dv.model_dump(), "_per_defendant": True},
                        )
                    keep = {n.name for n in named}
                    defendants = [d for d in defendants if d.name not in keep]
                    multi = len(defendants) > 1
                else:
                    # All accused acquitted → short-circuit to the disposition and end.
                    verdict = _directed_acquittal(defendants, case.case_type)
                    yield TrialEvent(phase="Verdict", speaker="System", kind="phase", content="The Verdict")
                    if len(defendants) > 1:
                        for dv in verdict.per_defendant:
                            yield TrialEvent(
                                phase="Verdict", speaker=f"Verdict — {dv.defendant_name}",
                                kind="structured", content=f"{dv.defendant_name}: {dv.outcome} (directed)",
                                data={**dv.model_dump(), "_per_defendant": True},
                            )
                    else:
                        yield TrialEvent(
                            phase="Verdict", speaker="Jury Foreperson", kind="structured",
                            content=f"Verdict: {verdict.outcome} (directed)", data=verdict.model_dump(),
                        )
                    disp = "dismiss the claim" if case.case_type == "civil" else "discharge the accused"
                    yield TrialEvent(phase="Ruling", speaker="System", kind="phase", content="The Judge's Ruling")
                    yield TrialEvent(
                        phase="Ruling", speaker="Judge", kind="structured",
                        content="The accused are discharged on a directed verdict.",
                        data=JudgeRuling(
                            verdict_acknowledgement="A directed verdict of acquittal has been entered.",
                            reasoning=dvr.reasoning,
                            sentence_or_remedy=f"As a directed verdict has been entered, the Court must {disp}.",
                            closing_remarks="Court is adjourned.",
                        ).model_dump(),
                    )
                    return

    # --- 5. Argument rounds ---------------------------------------------
    no_evidence_ref = "" if with_witnesses else " Do not reference witnesses or evidence."
    for rnd in range(1, case.argument_rounds + 1):
        yield TrialEvent(
            phase=f"Arguments R{rnd}",
            speaker="System",
            kind="phase",
            content=f"Argument Round {rnd} of {case.argument_rounds}",
        )
        yield await speak(
            crown,
            f"This is oral ARGUMENT round {rnd} of {case.argument_rounds} — not an "
            "opening or closing. BEGIN by stating, in its strongest and fairest form, the "
            "single best point the Defense has made so far (steelman it in a sentence or "
            "two), then rebut it head-on with the facts and the law. THEN advance the "
            f"{prosecutor}'s case and answer the defense's other recent points.\n\n"
            f"{crown_persona_text}\n\n{crown_strategy_text}\n\n{case_brief}\n\nTranscript so far:\n{transcript_text()}",
            phase=f"Arguments R{rnd}",
            speaker=prosecutor,
        )
        yield await speak(
            defense,
            f"This is oral ARGUMENT round {rnd} of {case.argument_rounds} — not an "
            "opening or closing. BEGIN by stating, in its strongest and fairest form, the "
            f"single best point the {prosecutor} has just made (steelman it in a sentence "
            "or two), then rebut it head-on, exposing the gap or the reasonable doubt. THEN "
            "advance the defense's theory.\n\n"
            f"{defense_persona_text}\n\n{defense_strategy_text}\n\n{case_brief}\n\nTranscript so far:\n{transcript_text()}",
            phase=f"Arguments R{rnd}",
            speaker="Defense",
        )
        if rnd < case.argument_rounds:
            yield await speak(
                judge,
                f"{judge_persona_text}\n\nBriefly interject to keep the proceedings focused, "
                f"then invite the next round of argument.{no_evidence_ref}\n\n"
                f"Transcript so far:\n{transcript_text()}",
                phase=f"Arguments R{rnd}",
                speaker="Judge",
            )

    # --- 6. Closing statements ------------------------------------------
    yield TrialEvent(
        phase="Closing", speaker="System", kind="phase", content="Closing Statements"
    )
    crown_close = await speak(
        crown,
        "Deliver your CLOSING STATEMENT. Squarely confront the single strongest point the "
        "Defense has made and explain why it does not raise a reasonable doubt, then tie "
        f"your theory together.\n\n{crown_persona_text}\n\n{crown_strategy_text}\n\n{case_brief}\n\n"
        f"Transcript so far:\n{transcript_text()}",
        phase="Closing",
        speaker=prosecutor,
        self_check=True,
    )
    yield crown_close
    fc = await ground_event(crown_close.content, speaker=prosecutor, phase="Closing", gphase="closing")
    if fc:
        yield fc
    defense_close = await speak(
        defense,
        "Deliver your CLOSING STATEMENT. Squarely confront the single strongest point the "
        f"{prosecutor} has made and explain why it falls short of the standard of proof, "
        f"then tie your theory together.\n\n{defense_persona_text}\n\n{defense_strategy_text}\n\n{case_brief}\n\n"
        f"Transcript so far:\n{transcript_text()}",
        phase="Closing",
        speaker="Defense",
        self_check=True,
    )
    yield defense_close
    fc = await ground_event(defense_close.content, speaker="Defense", phase="Closing", gphase="closing")
    if fc:
        yield fc

    # --- 7. Jury deliberation -------------------------------------------
    yield TrialEvent(
        phase="Deliberation", speaker="System", kind="phase", content="The Jury Deliberates"
    )
    full_transcript = transcript_text()

    async def get_vote(persona: JurorPersona, extra: str = "") -> JurorVote:
        per_def = ""
        if multi:
            names = ", ".join(d.name for d in defendants)
            per_def = (
                f"\n\nThere are MULTIPLE co-accused: {names}. Return one entry in "
                "'defendant_votes' for EACH of them, judged on that person's own role "
                "and conduct — they may receive different verdicts, each with its own "
                "per-element findings."
            )
        charge_note = ""
        if multi_c:
            cnames = ", ".join(c.label for c in charges)
            charge_note = (
                f"\n\nThere are MULTIPLE charges ({cnames}). Judge EACH charge separately on "
                "its own elements — the accused may be guilty on one charge and not another. "
                + (
                    "For each co-accused, return one 'charge_votes' entry per charge INSIDE "
                    "that accused's 'defendant_votes' entry."
                    if multi
                    else "Return one 'charge_votes' entry per charge (each with its own "
                         "per-element findings)."
                )
            )
        prompt = (
            f"Your persona — name: {persona.name}; background: {persona.background}; "
            f"disposition: {persona.disposition}.\n\n{case_brief}\n\n"
            f"Full trial transcript:\n{full_transcript}{per_def}{charge_note}{extra}"
        )
        try:
            vote = await run_structured(juror_agent, prompt, JurorVote, require_english=True)
            vote.juror_name = persona.name  # keep the persona name even if the model drifts
            # The verdict is driven by the per-element findings: convict only if every
            # essential element is proven. Falls back to the text/enum reading if the
            # juror gave no element findings. Applied at every level (top, per-charge,
            # per-defendant, per-defendant-per-charge).
            vote.vote = _derive_vote(vote.element_findings, vote.vote, vote.verdict)
            for cv in vote.charge_votes:
                cv.vote = _derive_vote(cv.element_findings, cv.vote, cv.verdict)
            for dv in vote.defendant_votes:
                dv.vote = _derive_vote(dv.element_findings, dv.vote, dv.verdict)
                for cv in dv.charge_votes:
                    cv.vote = _derive_vote(cv.element_findings, cv.vote, cv.verdict)
            return vote
        except Exception as exc:
            return JurorVote(
                juror_name=persona.name,
                verdict=acquit_word,
                vote="acquit",
                confidence=1,
                reasoning=f"(Could not deliberate: {exc})",
            )

    def tally(vs: list[JurorVote]) -> Verdict:
        return _tally(vs, defendants, charges, case.case_type)

    # ---- jury-room discussion (when deliberation_style == "dialogue") ----
    dialogue = case.deliberation_style == "dialogue" and len(personas) > 1
    deliberation_agent = build_deliberation_agent(model)
    foreperson_agent = build_foreperson_agent(model)
    foreperson = personas[0]  # the first empanelled juror chairs the room
    room: list[str] = []      # the running jury-room discussion across all rounds

    per_def_note = ""
    if multi:
        _names = ", ".join(d.name for d in defendants)
        per_def_note = (
            f"\n\nThere are MULTIPLE co-accused ({_names}); weigh each one separately."
        )

    def _delib_status(v: Verdict | None) -> str:
        if v is None:
            return ""
        if v.per_defendant:
            parts = "; ".join(
                d.defendant_name + ": " + ", ".join(f"{k} {n}" for k, n in d.tally.items())
                for d in v.per_defendant
            )
        else:
            parts = ", ".join(f"{k} {n}" for k, n in v.tally.items())
        return f"\nThe current vote split is — {parts}."

    async def foreperson_turn(opening: bool, v: Verdict | None) -> DeliberationRemark:
        room_text = "\n".join(room) or "(the room has not spoken yet)"
        if opening:
            ask = (
                "Open the jury's deliberations: frame the question, remind the room of the "
                "standard of proof and that they must work through EACH legal element for "
                "EACH accused, and invite discussion."
            )
            status = ""
        else:
            ask = (
                "Take stock: neutrally summarise where the room stands and what is still in "
                "dispute, and focus the next round on the unresolved elements."
            )
            status = _delib_status(v)
        prompt = (
            f"You are the JURY FOREPERSON. Your persona — name: {foreperson.name}; "
            f"background: {foreperson.background}; disposition: {foreperson.disposition}.\n\n"
            f"{case_brief}\n\nFull trial transcript:\n{full_transcript}\n\n"
            f"Jury-room discussion so far:\n{room_text}{status}\n\n{ask}"
        )
        try:
            r = await run_structured(foreperson_agent, prompt, DeliberationRemark, require_english=True)
            r.juror_name = foreperson.name
        except Exception as exc:
            r = DeliberationRemark(
                juror_name=foreperson.name,
                statement=f"(The foreperson was unable to speak: {exc})",
            )
        room.append(f"Foreperson {r.juror_name}: {r.statement}")
        return r

    async def juror_remark(persona: JurorPersona) -> DeliberationRemark:
        room_text = "\n".join(room) or "(no one has spoken yet — you may open the discussion)"
        prompt = (
            f"Your persona — name: {persona.name}; background: {persona.background}; "
            f"disposition: {persona.disposition}.\n\n{case_brief}\n\n"
            f"Full trial transcript:\n{full_transcript}\n\n"
            f"Jury-room discussion so far — respond to it:\n{room_text}{per_def_note}"
        )
        try:
            r = await run_structured(deliberation_agent, prompt, DeliberationRemark, require_english=True)
            r.juror_name = persona.name
        except Exception as exc:
            r = DeliberationRemark(juror_name=persona.name, statement=f"(unable to speak: {exc})")
        room.append(f"{r.juror_name}: {r.statement}")
        return r

    # --- 7a. Private straw poll BEFORE any discussion (anti-herding) ----
    # Each juror votes independently first, blind to the others — capturing their
    # own first read before social influence. Never threaded into later prompts.
    straw_votes: list[JurorVote] = []
    if case.straw_poll and len(personas) > 1:
        straw_extra = (
            "\n\nThis is a PRIVATE STRAW POLL taken BEFORE any discussion. Vote "
            "independently on your OWN reading of the elements — you have NOT heard the "
            "other jurors, so do not reference them or any discussion."
        )
        straw_votes = list(await asyncio.gather(*[get_vote(p, straw_extra) for p in personas]))
        yield TrialEvent(
            phase="Deliberation",
            speaker="Court Clerk",
            kind="structured",
            content="Initial straw poll — taken privately, before discussion.",
            data={"_straw": True, **tally(straw_votes).model_dump()},
        )

    votes: list[JurorVote] = []
    verdict: Verdict | None = None
    prev_sig = None
    for d_round in range(1, case.deliberation_rounds + 1):
        if d_round > 1:
            yield TrialEvent(
                phase="Deliberation",
                speaker="System",
                kind="phase",
                content=f"Deliberation round {d_round}: the jury reconsiders",
            )

        if dialogue:
            # The foreperson frames the round, then jurors speak in turn — each one
            # hearing what the room has said so far and replying to it. THEN they vote.
            fr = await foreperson_turn(opening=(d_round == 1), v=verdict)
            yield TrialEvent(
                phase="Deliberation",
                speaker=f"Jury Foreperson — {fr.juror_name}",
                kind="message",
                content=fr.statement,
            )
            for p in personas:
                rr = await juror_remark(p)
                lean = f"  [leaning: {rr.leaning}]" if rr.leaning else ""
                yield TrialEvent(
                    phase="Deliberation",
                    speaker=f"Juror — {rr.juror_name}",
                    kind="message",
                    content=f"{rr.statement}{lean}",
                )
            discussion = "\n".join(room)
            extra = (
                "\n\nThe jury has just discussed the case in the jury room. Here is that "
                f"discussion:\n{discussion}\n\nNow cast your formal vote for THIS round, "
                "consistent with your findings on each legal element."
            )
        elif d_round == 1:
            extra = ""
        else:
            others = "\n".join(
                f"- {v.juror_name} voted {v.verdict}: {v.reasoning}" for v in votes
            )
            tally_line = ", ".join(f"{k}: {n}" for k, n in verdict.tally.items())
            extra = (
                f"\n\nThis is deliberation round {d_round}. The running tally is "
                f"{tally_line}. Here is what the other jurors said:\n{others}\n\n"
                "Reconsider and give your vote for THIS round. Hold firm if the arguments "
                "justify it, or change your mind if you are genuinely persuaded."
            )

        votes = list(await asyncio.gather(*[get_vote(p, extra) for p in personas]))
        if d_round == 1:
            prefix = ""
        elif d_round == case.deliberation_rounds:
            prefix = "FINAL: "
        else:
            prefix = "(revised) "
        for v in votes:
            yield TrialEvent(
                phase="Deliberation",
                speaker=f"Juror — {v.juror_name}",
                kind="structured",
                content=f"{prefix}{v.verdict} (confidence {v.confidence}/10)",
                data=v.model_dump(),
            )
        verdict = tally(votes)
        # Exit early only if the verdict is settled AND the carrying side holds it
        # with real conviction — a shaky (low-confidence) consensus gets another
        # (bounded) round instead of ending immediately.
        if (_all_settled(verdict) and _settled_with_conviction(votes, verdict)) or len(personas) <= 1:
            break
        sig = _tally_signature(votes)
        if d_round > 1 and sig == prev_sig:
            break  # the divided jury has stopped moving — deadlocked
        prev_sig = sig

    # How the room moved from the private straw poll to the final vote.
    if straw_votes:
        yield TrialEvent(
            phase="Deliberation",
            speaker="Court Clerk",
            kind="structured",
            content="How the room moved after deliberation.",
            data={"_movement": True, **_straw_movement(straw_votes, votes, defendants, charges, case.case_type).model_dump()},
        )

    # --- 8. Verdict ------------------------------------------------------
    yield TrialEvent(phase="Verdict", speaker="System", kind="phase", content="The Verdict")
    if multi:
        for dv in verdict.per_defendant:
            yield TrialEvent(
                phase="Verdict",
                speaker=f"Verdict — {dv.defendant_name}",
                kind="structured",
                content=f"{dv.defendant_name}: {dv.outcome}",
                data={**dv.model_dump(), "_per_defendant": True},
            )
            # In a multi-charge case, the per-charge breakdown is the real verdict.
            for cv in dv.per_charge:
                yield TrialEvent(
                    phase="Verdict",
                    speaker=f"Verdict — {dv.defendant_name}: {cv.charge_label}",
                    kind="structured",
                    content=f"{dv.defendant_name} — {cv.charge_label}: {cv.outcome}",
                    data={**cv.model_dump(), "_per_charge": True, "defendant_name": dv.defendant_name},
                )
            charge_line = (
                "; ".join(f"{cv.charge_label}: {cv.outcome}" for cv in dv.per_charge)
                if dv.per_charge else dv.outcome
            )
            log("Jury Foreperson", f"As to {dv.defendant_name}: {charge_line}. {dv.dissent_summary}")
    elif verdict.per_charge:
        for cv in verdict.per_charge:
            yield TrialEvent(
                phase="Verdict",
                speaker=f"Verdict — {cv.charge_label}",
                kind="structured",
                content=f"{cv.charge_label}: {cv.outcome}",
                data={**cv.model_dump(), "_per_charge": True},
            )
        log("Jury Foreperson", "; ".join(f"{cv.charge_label}: {cv.outcome}" for cv in verdict.per_charge))
    else:
        yield TrialEvent(
            phase="Verdict",
            speaker="Jury Foreperson",
            kind="structured",
            content=f"Verdict: {verdict.outcome}",
            data=verdict.model_dump(),
        )
        log("Jury Foreperson", f"The jury finds: {verdict.outcome}. {verdict.dissent_summary}")

    # --- 9. Judge's ruling ----------------------------------------------
    yield TrialEvent(phase="Ruling", speaker="System", kind="phase", content="The Judge's Ruling")
    remedy = "remedy / damages" if case.case_type == "civil" else "sentence"
    disposition = "dismiss the claim" if case.case_type == "civil" else "discharge the accused"
    if multi:
        def _def_outcomes(dv):
            if dv.per_charge:
                inner = ", ".join(f"{cv.charge_label} {cv.outcome}" for cv in dv.per_charge)
                return f"{dv.defendant_name}: {inner}"
            return f"{dv.defendant_name}: {dv.outcome}"
        outcomes = "; ".join(_def_outcomes(dv) for dv in verdict.per_defendant)
        ruling_directive = (
            "The jury returned SEPARATE verdicts for each co-accused (and each charge): "
            + outcomes + ". "
            f"Address EACH co-accused and EACH charge individually — impose a fitting "
            f"{remedy} where convicted/liable, {disposition} where acquitted/not liable, "
            "and declare a mistrial on any charge or accused on whom the jury hung."
        )
        verdict_line = "The jury's verdicts are FINAL and binding on you: " + outcomes + "."
    elif verdict.per_charge:
        outcomes = "; ".join(f"{cv.charge_label}: {cv.outcome}" for cv in verdict.per_charge)
        ruling_directive = (
            "The jury returned SEPARATE verdicts on each charge: " + outcomes + ". Address "
            f"EACH charge — impose a fitting {remedy} on any charge of conviction/liability, "
            f"{disposition} on any acquittal, and declare a mistrial on any hung charge."
        )
        verdict_line = "The jury's verdicts are FINAL and binding on you: " + outcomes + "."
    else:
        if verdict.hung:
            ruling_directive = (
                "The jury is HUNG. Declare a mistrial and explain the next steps (the "
                f"{prosecutor} may seek a new trial or stay the matter). Do NOT convict, "
                "acquit, or decide the merits yourself."
            )
        elif verdict.outcome == convict_word:
            ruling_directive = (
                f"The jury found the accused {convict_word}. Impose a fitting {remedy}."
            )
        else:
            ruling_directive = f"The jury found the accused {acquit_word}. You must {disposition}."
        verdict_line = (
            f"The jury's verdict is FINAL and binding on you: {verdict.outcome}. "
            f"Tally: {verdict.tally}. {verdict.dissent_summary}"
        )
    ruling_prompt = (
        f"{judge_persona_text}\n\n{case_brief}\n\nFull trial transcript:\n{transcript_text()}\n\n"
        f"{verdict_line}\n\n{ruling_directive}\n\nDeliver your ruling."
    )
    ruling: JudgeRuling = await run_structured(
        ruling_agent, ruling_prompt, JudgeRuling, require_english=True
    )
    # The model sometimes leaves the headline disposition blank — most often on an
    # acquittal, where there is no sentence to impose. Supply a sensible fallback so
    # the ruling always leads with a disposition line rather than an empty bubble.
    if not ruling.sentence_or_remedy.strip():
        if multi or verdict.per_charge:
            ruling.sentence_or_remedy = "The Court's disposition on each count is set out above."
        elif verdict.hung:
            ruling.sentence_or_remedy = "A mistrial is declared."
        elif verdict.outcome == convict_word:
            ruling.sentence_or_remedy = f"A {remedy} is imposed."
        else:
            ruling.sentence_or_remedy = (
                "The claim is dismissed." if case.case_type == "civil"
                else "The accused is discharged."
            )
    yield TrialEvent(
        phase="Ruling",
        speaker="Judge",
        kind="structured",
        content=ruling.sentence_or_remedy,
        data=ruling.model_dump(),
    )
    # Fact-check the ruling's reasoning + disposition (multi-checker when adversarial).
    rfc = await ground_event(
        f"{ruling.reasoning} {ruling.sentence_or_remedy}",
        speaker="Judge", phase="Ruling", gphase="ruling",
        checkers=3 if case.grounding_adversarial else 1,
    )
    if rfc:
        yield rfc

"""Trial state machine. Streams a sequence of TrialEvent objects."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .agents import (
    build_crown_agent,
    build_defense_agent,
    build_intake_agent,
    build_judge_agent,
    build_juror_agent,
    build_juror_pool_agent,
    build_ruling_agent,
)
from .llm_utils import run_structured
from .model_factory import build_model
from .schemas import (
    CaseInput,
    JuryPool,
    JurorPersona,
    JurorVote,
    JudgeRuling,
    Speech,
    StructuredCase,
    TrialEvent,
    Verdict,
)


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------
def _is_acquittal(verdict_text: str) -> bool:
    """True if a juror's verdict favours the defendant (not guilty / not liable)."""
    s = verdict_text.strip().lower()
    return ("not" in s) or ("acquit" in s) or ("innocent" in s)


def _tally_votes(votes: list[JurorVote], case_type: str) -> Verdict:
    convict_word = "Liable" if case_type == "civil" else "Guilty"
    acquit_word = "Not Liable" if case_type == "civil" else "Not Guilty"

    convict = sum(1 for v in votes if not _is_acquittal(v.verdict))
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
            if (_is_acquittal(v.verdict) and outcome == convict_word)
            or (not _is_acquittal(v.verdict) and outcome == acquit_word)
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

    intake = build_intake_agent(model)
    crown = build_crown_agent(model)
    defense = build_defense_agent(model)
    judge = build_judge_agent(model)
    ruling_agent = build_ruling_agent(model)
    pool_agent = build_juror_pool_agent(model)
    juror_agent = build_juror_agent(model)

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

    async def speak(agent, prompt: str, *, phase: str, speaker: str) -> TrialEvent:
        """Run a free-form role and return its complete statement as a message event.

        The statement is fetched as JSON ({"statement": ...}) so the full text
        survives the model's <think> block; the browser reveals it with a
        typewriter effect for a live feel.
        """
        try:
            text = (await run_structured(agent, prompt, Speech)).statement.strip()
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
    sc: StructuredCase = await run_structured(intake, intake_prompt, StructuredCase)
    log("Court Clerk", f"Case filed: {sc.case_caption}. {sc.summary}")
    yield TrialEvent(
        phase="Intake",
        speaker="Law Firm Clerk",
        kind="structured",
        content=sc.summary,
        data=sc.model_dump(),
    )

    case_brief = (
        f"CASE: {sc.case_caption}\n"
        f"Type: {case.case_type} | Jurisdiction: {case.jurisdiction} | "
        f"Standard of proof: {standard}\n"
        f"Charges/Claims: {'; '.join(sc.charges_or_claims)}\n"
        f"Summary: {sc.summary}\n"
        f"Key facts: {'; '.join(sc.key_facts)}\n"
        f"Prosecution theory: {sc.prosecution_theory}\n"
        f"Defense theory: {sc.defense_theory}"
    )

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
    yield await speak(
        judge,
        f"Open this trial and instruct the jury.\n\n{case_brief}",
        phase="Opening",
        speaker="Judge",
    )

    # --- 4. Opening statements ------------------------------------------
    yield await speak(
        crown,
        f"Deliver your OPENING STATEMENT.\n\n{case_brief}",
        phase="Opening",
        speaker=prosecutor,
    )
    yield await speak(
        defense,
        f"Deliver your OPENING STATEMENT.\n\n{case_brief}\n\nTranscript so far:\n{transcript_text()}",
        phase="Opening",
        speaker="Defense",
    )

    # --- 5. Argument rounds ---------------------------------------------
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
            f"opening or closing. Advance the {prosecutor}'s case using only the agreed "
            "facts, and answer the defense's most recent points.\n\n"
            f"{case_brief}\n\nTranscript so far:\n{transcript_text()}",
            phase=f"Arguments R{rnd}",
            speaker=prosecutor,
        )
        yield await speak(
            defense,
            f"This is oral ARGUMENT round {rnd} of {case.argument_rounds} — not an "
            f"opening or closing. Respond directly to the {prosecutor}'s most recent "
            "argument, using only the agreed facts.\n\n"
            f"{case_brief}\n\nTranscript so far:\n{transcript_text()}",
            phase=f"Arguments R{rnd}",
            speaker="Defense",
        )
        if rnd < case.argument_rounds:
            yield await speak(
                judge,
                "Briefly interject to keep the proceedings focused, then invite the "
                "next round of argument. Do not reference witnesses or evidence.\n\n"
                f"Transcript so far:\n{transcript_text()}",
                phase=f"Arguments R{rnd}",
                speaker="Judge",
            )

    # --- 6. Closing statements ------------------------------------------
    yield TrialEvent(
        phase="Closing", speaker="System", kind="phase", content="Closing Statements"
    )
    yield await speak(
        crown,
        f"Deliver your CLOSING STATEMENT.\n\n{case_brief}\n\nTranscript so far:\n{transcript_text()}",
        phase="Closing",
        speaker=prosecutor,
    )
    yield await speak(
        defense,
        f"Deliver your CLOSING STATEMENT.\n\n{case_brief}\n\nTranscript so far:\n{transcript_text()}",
        phase="Closing",
        speaker="Defense",
    )

    # --- 7. Jury deliberation -------------------------------------------
    yield TrialEvent(
        phase="Deliberation", speaker="System", kind="phase", content="The Jury Deliberates"
    )
    full_transcript = transcript_text()

    async def get_vote(persona: JurorPersona, extra: str = "") -> JurorVote:
        prompt = (
            f"Your persona — name: {persona.name}; background: {persona.background}; "
            f"disposition: {persona.disposition}.\n\n{case_brief}\n\n"
            f"Full trial transcript:\n{full_transcript}{extra}"
        )
        try:
            vote = await run_structured(juror_agent, prompt, JurorVote)
            # Make sure the name matches the persona even if the model drifts.
            vote.juror_name = persona.name
            return vote
        except Exception as exc:
            return JurorVote(
                juror_name=persona.name,
                verdict=acquit_word,
                confidence=1,
                reasoning=f"(Could not deliberate: {exc})",
            )

    votes: list[JurorVote] = list(await asyncio.gather(*[get_vote(p) for p in personas]))
    for v in votes:
        yield TrialEvent(
            phase="Deliberation",
            speaker=f"Juror — {v.juror_name}",
            kind="structured",
            content=f"{v.verdict} (confidence {v.confidence}/10)",
            data=v.model_dump(),
        )

    verdict = _tally_votes(votes, case.case_type)

    # One revision round if the jury is split, so they can reconsider.
    if not verdict.unanimous and len(personas) > 1:
        yield TrialEvent(
            phase="Deliberation",
            speaker="System",
            kind="phase",
            content="The jury reconsiders after comparing views",
        )
        others = "\n".join(
            f"- {v.juror_name} voted {v.verdict}: {v.reasoning}" for v in votes
        )
        tally_line = ", ".join(f"{k}: {n}" for k, n in verdict.tally.items())
        extra = (
            f"\n\nThe jury's first vote stands at {tally_line}. Here is what the "
            f"other jurors said:\n{others}\n\nReconsider and give your FINAL vote. "
            "You may keep or change your position."
        )
        votes = list(await asyncio.gather(*[get_vote(p, extra) for p in personas]))
        for v in votes:
            yield TrialEvent(
                phase="Deliberation",
                speaker=f"Juror — {v.juror_name}",
                kind="structured",
                content=f"FINAL: {v.verdict} (confidence {v.confidence}/10)",
                data=v.model_dump(),
            )
        verdict = _tally_votes(votes, case.case_type)

    # --- 8. Verdict ------------------------------------------------------
    yield TrialEvent(phase="Verdict", speaker="System", kind="phase", content="The Verdict")
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
    if verdict.hung:
        ruling_directive = (
            "The jury is HUNG. Declare a mistrial and explain the next steps (the "
            f"{prosecutor} may seek a new trial or stay the matter). Do NOT convict, "
            "acquit, or decide the merits yourself."
        )
    elif verdict.outcome == convict_word:
        remedy = "remedy / damages" if case.case_type == "civil" else "sentence"
        ruling_directive = (
            f"The jury found the accused {convict_word}. Impose a fitting {remedy}."
        )
    else:
        disposition = (
            "dismiss the claim" if case.case_type == "civil" else "discharge the accused"
        )
        ruling_directive = f"The jury found the accused {acquit_word}. You must {disposition}."
    ruling_prompt = (
        f"{case_brief}\n\nFull trial transcript:\n{transcript_text()}\n\n"
        f"The jury's verdict is FINAL and binding on you: {verdict.outcome}. "
        f"Tally: {verdict.tally}. {verdict.dissent_summary}\n\n"
        f"{ruling_directive}\n\nDeliver your ruling."
    )
    ruling: JudgeRuling = await run_structured(ruling_agent, ruling_prompt, JudgeRuling)
    yield TrialEvent(
        phase="Ruling",
        speaker="Judge",
        kind="structured",
        content=ruling.sentence_or_remedy,
        data=ruling.model_dump(),
    )

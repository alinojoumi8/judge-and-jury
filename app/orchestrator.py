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
    build_witness_agent,
)
from .llm_utils import run_structured
from .model_factory import build_model
from .schemas import (
    CaseInput,
    Defendant,
    DefendantVerdict,
    ExaminationQuestion,
    JuryPool,
    JurorPersona,
    JurorVote,
    JudgeRuling,
    Objection,
    ObjectionRuling,
    Speech,
    StructuredCase,
    TrialEvent,
    Verdict,
    WitnessAnswer,
)


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


def _tally_votes_multi(
    votes: list[JurorVote], defendants: list[Defendant], case_type: str
) -> Verdict:
    """Tally a separate verdict for each co-accused from one set of juror ballots.

    Each juror returns all per-defendant votes in one response; we project those
    onto the single-defendant `_tally_votes` per accused. Missing/misspelled
    entries fall back to the juror's top-level vote.
    """
    per: list[DefendantVerdict] = []
    for d in defendants:
        projected: list[JurorVote] = []
        for v in votes:
            m = next(
                (x for x in v.defendant_votes
                 if x.defendant_name.strip().lower() == d.name.strip().lower()),
                None,
            )
            src = m if m is not None else v
            projected.append(
                JurorVote(
                    juror_name=v.juror_name,
                    verdict=src.verdict,
                    vote=src.vote,
                    confidence=src.confidence,
                    reasoning=src.reasoning,
                )
            )
        base = _tally_votes(projected, case_type)
        per.append(
            DefendantVerdict(
                defendant_name=d.name,
                role=d.role,
                **base.model_dump(exclude={"per_defendant"}),
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


def _tally_signature(votes: list[JurorVote]) -> tuple:
    """A hashable fingerprint of the jury's current position (per defendant if any).

    Used to stop deliberation early when a divided jury stops moving.
    """
    sig = []
    for v in votes:
        if v.defendant_votes:
            inner = tuple(
                sorted((dv.defendant_name.strip().lower(), dv.vote) for dv in v.defendant_votes)
            )
            sig.append((v.juror_name, inner))
        else:
            sig.append((v.juror_name, v.vote))
    return tuple(sorted(sig, key=lambda x: x[0]))


def _all_settled(verdict: Verdict) -> bool:
    """True if there is nothing left to deliberate (every verdict is unanimous)."""
    if verdict.per_defendant:
        return all(d.unanimous for d in verdict.per_defendant)
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
        """Run a free-form role and return its complete statement as a message event."""
        try:
            text = (
                await run_structured(agent, prompt, Speech, require_english=True)
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

    defendants = _defendant_roster(case, sc)
    multi = len(defendants) > 1

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
    if multi:
        roster = "\n".join(
            f"- {d.name} ({d.role or 'role unstated'}): {d.account}" for d in defendants
        )
        case_brief += (
            "\n\nCO-ACCUSED (each must receive a SEPARATE verdict, judged on their "
            f"own role and conduct):\n{roster}"
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

    # --- 4b. Witness testimony & cross-examination (optional) -----------
    if with_witnesses:
        witness_agent = build_witness_agent(model)

        async def examine(examiner, examiner_name, objector, objector_name, w, witness_brief, kind):
            qa_log: list[str] = []
            yield TrialEvent(
                phase="Evidence", speaker="Court Clerk", kind="message",
                content=f"{kind.title()} examination of {w.name} by {examiner_name}.",
            )
            for i in range(1, case.qa_exchanges + 1):
                try:
                    q = (await run_structured(
                        examiner,
                        f"{kind} EXAMINATION of witness {w.name} ({w.role}). Ask question "
                        f"{i} of {case.qa_exchanges}. "
                        + ("Build your case." if kind == "DIRECT" else "Test or impeach the witness.")
                        + '\nReply with ONLY {"question": "your question"}.\n\n'
                        + f"{case_brief}\n\nExchange so far:\n" + ("\n".join(qa_log) or "(none)"),
                        ExaminationQuestion, require_english=True,
                    )).question.strip()
                except Exception as exc:
                    q = f"(No question put: {exc})"
                qa_log.append(f"{examiner_name}: {q}")
                yield TrialEvent(
                    phase="Evidence", speaker=examiner_name, kind="message",
                    content=q, data={"witness": w.name, "exam": kind},
                )

                # One deterministic objection on the first cross-examination question.
                if kind == "CROSS" and i == 1:
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
                            qa_log.append("(Objection sustained; the question is withdrawn.)")
                            continue

                try:
                    a = (await run_structured(
                        witness_agent,
                        f"You are {w.name} on the stand under {kind} examination. Answer "
                        f'truthfully, IN CHARACTER, only from what you know. Question: "{q}".'
                        f"\n\n{witness_brief}",
                        WitnessAnswer, require_english=True,
                    )).statement.strip()
                except Exception as exc:
                    a = f"(The witness could not answer: {exc})"
                qa_log.append(f"{w.name}: {a}")
                log(f"Witness {w.name}", a)
                yield TrialEvent(
                    phase="Evidence", speaker=f"Witness — {w.name}", kind="message",
                    content=a, data={"witness": w.name},
                )

        yield TrialEvent(
            phase="Evidence", speaker="System", kind="phase",
            content="Witness Testimony & Cross-Examination",
        )
        for w in case.witnesses[: case.max_witnesses]:
            if w.called_by == "prosecution":
                caller, caller_name, opp, opp_name = crown, prosecutor, defense, "Defense"
            else:
                caller, caller_name, opp, opp_name = defense, "Defense", crown, prosecutor
            witness_brief = (
                f"You are {w.name}, a {w.role} witness called by the {w.called_by}.\n"
                f"What you know:\n{w.what_they_know}\n\n{case_brief}"
            )
            yield TrialEvent(
                phase="Evidence", speaker="Court Clerk", kind="message",
                content=f"{w.name} is called to the stand ({w.role}, for the {w.called_by}).",
            )
            async for ev in examine(caller, caller_name, opp, opp_name, w, witness_brief, "DIRECT"):
                yield ev
            async for ev in examine(opp, opp_name, caller, caller_name, w, witness_brief, "CROSS"):
                yield ev

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
            f"opening or closing. Advance the {prosecutor}'s case, drawing on the agreed "
            "facts and any testimony on the record, and answer the defense's most recent "
            f"points.\n\n{case_brief}\n\nTranscript so far:\n{transcript_text()}",
            phase=f"Arguments R{rnd}",
            speaker=prosecutor,
        )
        yield await speak(
            defense,
            f"This is oral ARGUMENT round {rnd} of {case.argument_rounds} — not an "
            f"opening or closing. Respond directly to the {prosecutor}'s most recent "
            "argument, drawing on the agreed facts and any testimony on the record.\n\n"
            f"{case_brief}\n\nTranscript so far:\n{transcript_text()}",
            phase=f"Arguments R{rnd}",
            speaker="Defense",
        )
        if rnd < case.argument_rounds:
            yield await speak(
                judge,
                "Briefly interject to keep the proceedings focused, then invite the "
                f"next round of argument.{no_evidence_ref}\n\n"
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
        per_def = ""
        if multi:
            names = ", ".join(d.name for d in defendants)
            per_def = (
                f"\n\nThere are MULTIPLE co-accused: {names}. Return one entry in "
                "'defendant_votes' for EACH of them, judged on that person's own role "
                "and conduct — they may receive different verdicts."
            )
        prompt = (
            f"Your persona — name: {persona.name}; background: {persona.background}; "
            f"disposition: {persona.disposition}.\n\n{case_brief}\n\n"
            f"Full trial transcript:\n{full_transcript}{per_def}{extra}"
        )
        try:
            vote = await run_structured(juror_agent, prompt, JurorVote, require_english=True)
            vote.juror_name = persona.name  # keep the persona name even if the model drifts
            vote.vote = _normalize_vote(vote.verdict, vote.vote)
            for dv in vote.defendant_votes:
                dv.vote = _normalize_vote(dv.verdict, dv.vote)
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
        return (
            _tally_votes_multi(vs, defendants, case.case_type)
            if multi
            else _tally_votes(vs, case.case_type)
        )

    votes: list[JurorVote] = []
    verdict: Verdict | None = None
    prev_sig = None
    for d_round in range(1, case.deliberation_rounds + 1):
        if d_round == 1:
            extra = ""
        else:
            yield TrialEvent(
                phase="Deliberation",
                speaker="System",
                kind="phase",
                content=f"Deliberation round {d_round}: the jury reconsiders",
            )
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
        if _all_settled(verdict) or len(personas) <= 1:
            break
        sig = _tally_signature(votes)
        if d_round > 1 and sig == prev_sig:
            break  # the divided jury has stopped moving — deadlocked
        prev_sig = sig

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
            log("Jury Foreperson", f"As to {dv.defendant_name}: {dv.outcome}. {dv.dissent_summary}")
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
    if multi:
        outcomes = "; ".join(f"{dv.defendant_name}: {dv.outcome}" for dv in verdict.per_defendant)
        remedy = "remedy / damages" if case.case_type == "civil" else "sentence"
        disposition = "dismiss the claim" if case.case_type == "civil" else "discharge the accused"
        ruling_directive = (
            "The jury returned SEPARATE verdicts for each co-accused: " + outcomes + ". "
            f"Address EACH co-accused individually — impose a fitting {remedy} where "
            f"convicted/liable, {disposition} where acquitted/not liable, and declare a "
            "mistrial as to any defendant on whom the jury hung."
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
            remedy = "remedy / damages" if case.case_type == "civil" else "sentence"
            ruling_directive = (
                f"The jury found the accused {convict_word}. Impose a fitting {remedy}."
            )
        else:
            disposition = (
                "dismiss the claim" if case.case_type == "civil" else "discharge the accused"
            )
            ruling_directive = f"The jury found the accused {acquit_word}. You must {disposition}."
        verdict_line = (
            f"The jury's verdict is FINAL and binding on you: {verdict.outcome}. "
            f"Tally: {verdict.tally}. {verdict.dissent_summary}"
        )
    ruling_prompt = (
        f"{case_brief}\n\nFull trial transcript:\n{transcript_text()}\n\n"
        f"{verdict_line}\n\n{ruling_directive}\n\nDeliver your ruling."
    )
    ruling: JudgeRuling = await run_structured(
        ruling_agent, ruling_prompt, JudgeRuling, require_english=True
    )
    yield TrialEvent(
        phase="Ruling",
        speaker="Judge",
        kind="structured",
        content=ruling.sentence_or_remedy,
        data=ruling.model_dump(),
    )

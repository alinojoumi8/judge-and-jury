"""Offline end-to-end tests for the trial state machine — no network, no API key.

The unit tests in test_logic.py cover the pure helpers. These drive `run_trial`
itself with a scripted stand-in for the model, so the wiring between phases —
intake → digest → straw poll → deliberation → verdict → ruling — is exercised on
every test run rather than only on a paid trial.

Run: pytest -q
"""

from __future__ import annotations

import asyncio
from typing import Callable

import pytest

import app.orchestrator as orch
from app.schemas import (
    AgreedRecord,
    CaseInput,
    CaseStrategy,
    Charge,
    ChargeEvidence,
    ChargeVote,
    DefendantVote,
    DeliberationRemark,
    DirectedVerdictMotion,
    DirectedVerdictRuling,
    ElementEvidence,
    ElementFinding,
    EvidenceDigest,
    ExaminationQuestion,
    GroundingReport,
    JudgeRuling,
    JurorPersona,
    JurorVote,
    JuryPool,
    Objection,
    ObjectionRuling,
    RolePersona,
    Speech,
    StructuredCase,
    TrialCast,
    WitnessAnswer,
)

CHARGES = [
    Charge(label="Fraud over $5,000 (s.380(1)(a))", elements=[
        "a dishonest act", "deprivation caused by that act", "subjective knowledge of dishonesty",
    ]),
    Charge(label="Possession of proceeds of crime (s.354(1))", elements=[
        "possession of property", "the property was derived from a crime",
    ]),
]
DEFENDANTS = ["Alpha Adams", "Beta Brooks"]
JURY = 5


def _ballot(guilty: bool) -> JurorVote:
    """A deliberately messy ballot: short names, charge labels without the citation,
    and probabilities that straddle the 90% criminal threshold."""
    verdict = "guilty" if guilty else "not guilty"
    vote = "convict" if guilty else "acquit"
    return JurorVote(
        juror_name="drifted-name", verdict=verdict, vote=vote, confidence=8, reasoning="r",
        defendant_votes=[
            DefendantVote(
                defendant_name=d.split()[-1], verdict=verdict, vote=vote,
                charge_votes=[
                    ChargeVote(
                        charge_label=c.label.split(" (")[0], verdict=verdict, vote=vote,
                        element_findings=[
                            ElementFinding(element=e, proven=guilty, probability=95 if guilty else 30)
                            for e in c.elements
                        ],
                    )
                    for c in CHARGES
                ],
            )
            for d in DEFENDANTS
        ],
    )


def _install_fakes(monkeypatch, juror: Callable[[int], JurorVote]) -> dict:
    """Replace the model layer with a scripted one; return a call counter."""
    counts = {"juror": 0}

    async def fake_run_structured(agent, prompt, model_cls, retries=2, *, require_english=False):
        await asyncio.sleep(0)
        if model_cls is StructuredCase:
            return StructuredCase(
                case_caption="R. v. Adams & Brooks", charges_or_claims=[c.label for c in CHARGES],
                summary="s", key_facts=["f"], prosecution_theory="p", defense_theory="d",
                elements=CHARGES[0].elements, charges=CHARGES,
                agreed_record=AgreedRecord(parties=DEFENDANTS, figures=["raised: $1M"]),
            )
        if model_cls is CaseStrategy:
            return CaseStrategy(theory="t", strongest_points=["a"], opponents_best_point="b", rebuttal="c")
        if model_cls is TrialCast:
            return TrialCast(crown=RolePersona(name="C"), defense=RolePersona(name="D"),
                             judge=RolePersona(name="J"), witnesses=[RolePersona(name="Wanda Wu")])
        if model_cls is JuryPool:
            return JuryPool(jurors=[JurorPersona(name=f"Juror {i}", background="b", disposition="d")
                                    for i in range(JURY)])
        if model_cls is EvidenceDigest:
            return EvidenceDigest(charges=[
                ChargeEvidence(charge_label=c.label, elements=[
                    ElementEvidence(element=e, supporting=["s"], undermining=["u"], gaps=["g"])
                    for e in c.elements])
                for c in CHARGES])
        if model_cls is JurorVote:
            counts["juror"] += 1
            return juror(counts["juror"])
        if model_cls is DeliberationRemark:
            return DeliberationRemark(juror_name="x", statement="I think so.", leaning="acquit")
        if model_cls is ExaminationQuestion:
            return ExaminationQuestion(question="Where were you?")
        if model_cls is WitnessAnswer:
            return WitnessAnswer(statement="At home.")
        if model_cls is Objection:
            return Objection(object=False)
        if model_cls is ObjectionRuling:
            return ObjectionRuling(ruling="overruled", text="")
        if model_cls is DirectedVerdictMotion:
            return DirectedVerdictMotion(move=False)
        if model_cls is DirectedVerdictRuling:
            return DirectedVerdictRuling(granted=False)
        if model_cls is GroundingReport:
            return GroundingReport(grounded=True)
        if model_cls is JudgeRuling:
            # Fills sentencing detail regardless of outcome: the ruling guard must
            # strip it whenever nothing was proved.
            return JudgeRuling(
                verdict_acknowledgement="ack", reasoning="because", sentence_or_remedy="",
                aggravating_factors=["breach of trust"], sentencing_range="2-5 years",
                conditions=["no contact"],
            )
        if model_cls is Speech:
            return Speech(statement="I say words.")
        raise AssertionError(f"unhandled model class {model_cls}")

    monkeypatch.setattr(orch, "run_structured", fake_run_structured)
    monkeypatch.setattr(orch, "build_model", lambda name=None: object())
    for name in dir(orch):
        if name.startswith("build_") and name.endswith("_agent"):
            monkeypatch.setattr(orch, name, lambda model, **kw: object())
    return counts


def _case(**overrides) -> CaseInput:
    base = dict(
        title="Offline", case_type="criminal", charge_or_claim="fraud", your_side="a story",
        jury_size=JURY, argument_rounds=1, deliberation_rounds=2, qa_exchanges=1,
        defendants=[{"name": d, "role": "r", "account": "a"} for d in DEFENDANTS],
        witnesses=[{"name": "Wanda Wu", "role": "expert", "called_by": "prosecution",
                    "what_they_know": "things"}],
    )
    base.update(overrides)
    return CaseInput(**base)


def _run(case: CaseInput) -> list:
    async def go():
        return [ev async for ev in orch.run_trial(case)]
    return asyncio.run(go())


def _marker(ev) -> str:
    return next((k for k in (ev.data or {}) if k.startswith("_")), "")


def _verdicts(events) -> dict[str, str]:
    return {
        ev.speaker: ev.data["outcome"]
        for ev in events
        if ev.phase == "Verdict" and ev.kind == "structured" and "outcome" in (ev.data or {})
    }


# ---------------------------------------------------------------------------
def test_full_trial_wires_every_phase_and_never_errors(monkeypatch):
    # Every third ballot acquits, so the room is split and the trial must run to
    # the end of its rounds without an error event.
    counts = _install_fakes(monkeypatch, lambda n: _ballot(n % 3 != 0))
    events = _run(_case())

    assert not [e for e in events if e.kind == "error"], "trial raised"
    markers = {_marker(e) for e in events}
    assert {"_manifest", "_digest", "_straw", "_movement", "_per_defendant", "_per_charge"} <= markers
    phases = [e.content for e in events if e.kind == "phase"]
    assert "The Evidence, Element by Element" in phases and "The Verdict" in phases
    # straw poll + two rounds, one sample each
    assert counts["juror"] == JURY * 3


def test_short_names_and_bare_charge_labels_still_tally_per_charge(monkeypatch):
    # Ballots say "Adams" and "Fraud over $5,000"; the roster says "Alpha Adams" and
    # "Fraud over $5,000 (s.380(1)(a))". Nothing may fall back to the top-level vote.
    _install_fakes(monkeypatch, lambda n: _ballot(True))
    events = _run(_case())
    verdicts = _verdicts(events)
    per_charge = {k: v for k, v in verdicts.items() if ": " in k}
    assert len(per_charge) == len(DEFENDANTS) * len(CHARGES)
    assert set(per_charge.values()) == {"Guilty"}
    diag = [e for e in events if _marker(e) == "_diagnostics"]
    assert not diag or diag[0].data["unmatched_entries"] == []


def test_ruling_guard_strips_sentencing_when_nothing_was_proved(monkeypatch):
    _install_fakes(monkeypatch, lambda n: _ballot(n % 3 != 0))  # split -> hung
    events = _run(_case())
    ruling = next(e for e in events if e.phase == "Ruling" and e.kind == "structured")
    assert ruling.data["aggravating_factors"] == []
    assert ruling.data["sentencing_range"] == "" and ruling.data["conditions"] == []


def test_ruling_keeps_sentencing_on_a_conviction(monkeypatch):
    _install_fakes(monkeypatch, lambda n: _ballot(True))
    events = _run(_case())
    assert set(_verdicts(events).values()) == {"Guilty"}
    ruling = next(e for e in events if e.phase == "Ruling" and e.kind == "structured")
    assert ruling.data["aggravating_factors"] == ["breach of trust"]


def test_verdict_passes_takes_the_modal_ballot_and_reports_agreement(monkeypatch):
    # Samples alternate convict/convict/acquit: the modal ballot is convict, and each
    # juror should report 2/3 self-agreement on the deciding round.
    counts = _install_fakes(monkeypatch, lambda n: _ballot(n % 3 != 0))
    events = _run(_case(verdict_passes=3, deliberation_style="poll", deliberation_rounds=1))
    diag = next(e for e in events if _marker(e) == "_diagnostics")
    assert diag.data["verdict_passes"] == 3
    assert diag.data["ballots_counted"] == JURY
    assert 0 < diag.data["mean_sample_agreement"] < 1
    # straw poll (1 sample each) + one round at 3 samples each
    assert counts["juror"] == JURY + JURY * 3


def test_deadlock_exhortation_fires_exactly_once(monkeypatch):
    # Each juror holds the same position every round (3-2 split) -> the room stalls
    # -> the judge sends them back once -> a second stall is a genuine hung jury.
    counts = _install_fakes(monkeypatch, lambda n: _ballot(((n - 1) % JURY) < 3))
    events = _run(_case(deliberation_style="poll", deliberation_rounds=2))
    assert [e.content for e in events if e.kind == "phase" and "Deadlock" in e.content] == [
        "The Jury Reports a Deadlock"
    ]
    assert sum(1 for e in events if e.phase == "Deliberation" and e.speaker == "Judge") == 1
    # straw + round 1 + round 2 (stall) + the one extra round after the exhortation
    assert counts["juror"] == JURY * 4
    assert all(v.startswith("Hung jury") for v in _verdicts(events).values())


def test_no_extra_round_when_exhortation_is_off(monkeypatch):
    counts = _install_fakes(monkeypatch, lambda n: _ballot(((n - 1) % JURY) < 3))
    _run(_case(deliberation_style="poll", deliberation_rounds=2, deadlock_exhortation=False))
    assert counts["juror"] == JURY * 3


def test_failed_ballots_abstain_and_are_reported(monkeypatch):
    # Juror #2's ballot fails every time; the tally must exclude them rather than
    # count a phantom acquittal, and the diagnostics must name them.
    def juror(n):
        if (n - 1) % JURY == 1:
            raise RuntimeError("model down")
        return _ballot(True)

    _install_fakes(monkeypatch, juror)
    events = _run(_case(deliberation_style="poll", deliberation_rounds=1))
    diag = next(e for e in events if _marker(e) == "_diagnostics")
    assert diag.data["ballots_counted"] == JURY - 1
    assert diag.data["abstentions"] == ["Juror 1"]
    # Four counted convictions and one abstention is a unanimous conviction.
    assert set(_verdicts(events).values()) == {"Guilty"}


def test_digest_can_be_switched_off(monkeypatch):
    _install_fakes(monkeypatch, lambda n: _ballot(True))
    events = _run(_case(evidence_digest=False))
    assert not [e for e in events if _marker(e) == "_digest"]
    assert "The Evidence, Element by Element" not in [e.content for e in events if e.kind == "phase"]


def test_calibration_downgrades_an_overconfident_proven(monkeypatch):
    # Every element "proven" but scored 70%: below the 90% criminal threshold, so
    # the accused must be acquitted; with calibration off the boolean stands.
    def shaky(n):
        b = _ballot(True)
        for dv in b.defendant_votes:
            for cv in dv.charge_votes:
                for f in cv.element_findings:
                    f.probability = 70
        return b

    _install_fakes(monkeypatch, shaky)
    assert set(_verdicts(_run(_case(deliberation_rounds=1))).values()) == {"Not Guilty"}
    _install_fakes(monkeypatch, shaky)
    assert set(_verdicts(_run(_case(deliberation_rounds=1, calibrated_proof=False))).values()) == {"Guilty"}


@pytest.mark.parametrize("case_type,expected", [("criminal", "Guilty"), ("civil", "Liable")])
def test_case_type_words_flow_through_to_the_verdict(monkeypatch, case_type, expected):
    _install_fakes(monkeypatch, lambda n: _ballot(True))
    events = _run(_case(case_type=case_type, deliberation_rounds=1))
    assert set(_verdicts(events).values()) == {expected}

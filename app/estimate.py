"""How many model calls a trial configuration will cost — before it runs.

Cost scales as jury x passes x rounds, plus witnesses x exchanges x 3, and nothing
used to tell the user that a 12-juror, 5-witness, 3-pass trial is ~200 calls until
they had paid for it. This mirrors the phases in `orchestrator.run_trial` call for
call; keep the two in step when a phase changes.

Retries (parse failures, transport backoff) are not counted — they are the
exception, not the plan.
"""

from __future__ import annotations

from .schemas import CaseInput


def estimate_calls(case: CaseInput) -> dict:
    """Return {"min_calls", "max_calls", "breakdown"} for one run of `case`.

    `min_calls` assumes the happiest path: no objections drawn, no directed-verdict
    ruling needed, and a jury that settles unanimously in its first round.
    `max_calls` assumes every optional call fires and every round runs, including
    the one extra round after a judge's exhortation.
    """
    jury = case.jury_size
    passes = case.verdict_passes
    witnesses = min(len(case.witnesses), case.max_witnesses)
    grounding = case.grounding_check
    g_phases = set(case.grounding_phases) if grounding else set()

    b: dict[str, tuple[int, int]] = {}  # phase -> (min, max)

    # Intake, two case theories, casting, jury pool, judge's opening, two openings.
    setup = 1 + 2 + (1 if case.personas else 0) + 1 + 1 + 2
    b["setup"] = (setup, setup)

    # Each exchange: question + objection check + answer, and possibly a ruling on
    # the objection. Direct and cross each run qa_exchanges; re-direct qa_redirect.
    if witnesses:
        per_witness = 2 * case.qa_exchanges + (case.qa_redirect if case.redirect else 0)
        exchanges = witnesses * per_witness
        per_exchange_min = 3 + (1 if "witness" in g_phases else 0)
        b["evidence"] = (exchanges * per_exchange_min, exchanges * (per_exchange_min + 1))
        # The defence's motion, and the judge's ruling if it is actually made.
        if case.allow_directed_verdict:
            b["directed_verdict"] = (1, 2)

    # Two speeches per argument round, and a judicial interjection between rounds.
    args = 2 * case.argument_rounds + max(0, case.argument_rounds - 1)
    b["arguments"] = (args, args)

    closings = 2 * (2 if case.self_ground else 1) + (2 if "closing" in g_phases else 0)
    b["closings"] = (closings, closings)

    if case.evidence_digest:
        b["digest"] = (1, 1)

    if case.straw_poll and jury > 1:
        b["straw_poll"] = (jury, jury)

    # One deliberation round: in dialogue style the foreperson speaks and every
    # juror speaks before anyone votes; every juror then casts `passes` ballots.
    dialogue = case.deliberation_style == "dialogue" and jury > 1
    per_round = (1 + jury if dialogue else 0) + jury * passes
    max_rounds = case.deliberation_rounds + (1 if case.deadlock_exhortation else 0)
    exhortation = 1 if case.deadlock_exhortation else 0
    b["deliberation"] = (per_round, per_round * max_rounds + exhortation)

    ruling = 1 + ((3 if case.grounding_adversarial else 1) if "ruling" in g_phases else 0)
    b["ruling"] = (ruling, ruling)

    return {
        "min_calls": sum(lo for lo, _ in b.values()),
        "max_calls": sum(hi for _, hi in b.values()),
        "breakdown": {k: {"min": lo, "max": hi} for k, (lo, hi) in b.items()},
    }

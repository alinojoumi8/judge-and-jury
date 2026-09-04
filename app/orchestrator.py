"""Trial state machine. Streams a sequence of TrialEvent objects."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import AsyncIterator

from .agents import (
    build_caster_agent,
    build_crown_agent,
    build_defense_agent,
    build_deliberation_agent,
    build_digest_agent,
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
from .config import get_settings
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
    EvidenceDigest,
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
    RunManifest,
    Speech,
    StrawMovement,
    StructuredCase,
    TrialCast,
    TrialEvent,
    Verdict,
    WitnessAnswer,
)

logger = logging.getLogger(__name__)


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
# Label / element matching
#
# A juror's ballot is free text produced by a model: it paraphrases charge labels
# ("Fraud over $5,000" for "Fraud over $5,000 (s.380(1)(a))"), shortens names
# ("Marlowe" for "Dana Marlowe"), and rewords elements. Matching those
# by exact string silently drops the entry and falls back to a different vote — a
# wrong verdict that looks perfectly well-formed. So we match on meaning instead.
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")
# Words too common to carry any signal when comparing two legal phrasings.
_STOP = frozenset(
    "the a an of to that this and or in on at by for with as is was be been are it "
    "its their his her from any all not no s".split()
)
# Similarity above which two labels/elements are treated as the same thing. Set by
# hand against real ballots: comfortably above unrelated elements (~0.0-0.2) and
# below genuine rewordings of the same element (~0.5-0.9).
_MATCH_FLOOR = 0.45


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def _similarity(a: str, b: str) -> float:
    """0.0-1.0 similarity between two labels, tolerant of paraphrase and detail.

    Uses the overlap coefficient as well as Jaccard so that a short form fully
    contained in a longer one ("Marlowe" in "Dana Marlowe", "Fraud
    over $5,000" in "Fraud over $5,000 (s.380(1)(a))") scores as a match, while
    two genuinely different elements still score near zero.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if (a or "").strip().lower() == (b or "").strip().lower() else 0.0
    inter = len(ta & tb)
    if not inter:
        return 0.0
    jaccard = inter / len(ta | tb)
    overlap = inter / min(len(ta), len(tb))
    return max(jaccard, 0.9 * overlap)


def _pick_by_label(entries: list, attr: str, target: str):
    """The entry whose `attr` best names `target`, or None if nothing is close.

    Returning None (rather than a bad guess) is what lets the caller record an
    honest fallback instead of quietly tallying the wrong vote.
    """
    best, best_score = None, 0.0
    for e in entries:
        score = _similarity(target, getattr(e, attr, "") or "")
        if score > best_score:
            best, best_score = e, score
    return best if best_score >= _MATCH_FLOOR else None


def _element_proven(finding, threshold: int | None = None) -> bool:
    """Is this element established to the standard of proof?

    The juror's boolean is necessary but not sufficient: when they also gave a
    probability, it has to actually meet the standard they were charged with. This
    only ever DOWNGRADES an over-confident "proven" — it never turns a "not proven"
    into a finding against the accused.
    """
    if not getattr(finding, "proven", False):
        return False
    p = getattr(finding, "probability", None)
    if threshold is not None and p is not None and p < threshold:
        return False
    return True


def _align_findings(required: list[str], findings: list) -> list:
    """Line each required element up with the juror's finding on it (None if absent).

    Greedy best-match, each finding used once. If nothing matches by wording but the
    juror returned exactly as many findings as there are elements, they almost
    certainly answered in order — so fall back to position rather than to nothing.
    """
    remaining = list(findings)
    out: list = []
    for req in required:
        best, best_score = None, 0.0
        for f in remaining:
            score = _similarity(req, getattr(f, "element", "") or "")
            if score > best_score:
                best, best_score = f, score
        if best is not None and best_score >= _MATCH_FLOOR:
            remaining.remove(best)
            out.append(best)
        else:
            out.append(None)
    # A juror who covered every element but reworded some of them beyond recognition
    # would otherwise be read as having left those elements unaddressed — an
    # acquittal on a wording quirk. When the leftovers exactly fill the holes, the
    # juror answered in order, so use position.
    holes = [i for i, x in enumerate(out) if x is None]
    if holes and len(remaining) == len(holes):
        for i, f in zip(holes, remaining):
            out[i] = f
    return out


def _proof_threshold(case: CaseInput) -> int:
    """The percentage at which an element counts as proven for this case type."""
    if case.proof_threshold is not None:
        return case.proof_threshold
    # ~90% for "beyond a reasonable doubt"; a bare majority for "balance of
    # probabilities". Both are conventions — override with `proof_threshold`.
    return 51 if case.case_type == "civil" else 90


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


def _derive_vote(
    findings: list,
    model_vote: str,
    verdict_text: str,
    *,
    required: list[str] | None = None,
    strict: bool = True,
    threshold: int | None = None,
) -> str:
    """Turn per-element findings into a 'convict'/'acquit' signal for tallying.

    When the juror worked through the legal elements, those are authoritative: the
    accused is convicted ONLY if every element is proven — a single unproven element
    means acquit. With no element findings, fall back to the free-text/enum reading.

    `required` is the authoritative element list from the charge. Supplying it fixes
    two ways a raw `all(proven)` gets the verdict wrong: a juror who addresses only
    some elements would otherwise convict on partial findings, and a juror who
    invents an extra element would otherwise acquit on it. With `strict`, an element
    the juror never addressed counts as NOT proven — the burden sits with the
    prosecution, so silence can never be a finding in its favour.
    """
    if not findings:
        return _normalize_vote(verdict_text, model_vote)
    if required:
        for f in _align_findings(required, findings):
            if f is None:
                if strict:
                    return "acquit"
                continue
            if not _element_proven(f, threshold):
                return "acquit"
        return "convict"
    return "convict" if all(_element_proven(f, threshold) for f in findings) else "acquit"


def _headline_from_charges(charge_votes: list) -> str:
    """A ballot's headline vote where it carries per-charge votes: convicted on any count.

    The per-charge votes are the real verdict in a multi-charge case, and each of
    them has been re-derived from element findings and the standard of proof. The
    headline used to come from the juror's free-text verdict instead, so it could
    say "guilty" over a breakdown that acquitted on every count.
    """
    return "convict" if any(cv.vote == "convict" for cv in charge_votes) else "acquit"


def _settled_with_conviction(votes: list[JurorVote], verdict, min_conf: int = 6) -> bool:
    """True only if the side that carried the verdict holds it with real conviction.

    An extra gate on early-exit: a unanimous/majority outcome held with LOW average
    confidence should get another (bounded) deliberation round rather than ending
    immediately. A genuinely confident room exits. `verdict` is accepted for API
    symmetry; the carrying side is read from the ballots so it works pre-tally too.
    """
    cast = [v for v in votes if not getattr(v, "abstained", False)]
    if not cast:
        return True
    convict_n = sum(1 for v in cast if v.vote == "convict")
    carrying = "convict" if convict_n * 2 > len(cast) else "acquit"
    side = [v for v in cast if v.vote == carrying]
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

    # A juror whose ballot could not be obtained ABSTAINS — they are not counted as
    # voting either way. Counting a failed model call as an acquittal (the old
    # behaviour) let one parse error force a hung jury in a criminal trial.
    cast = [v for v in votes if not getattr(v, "abstained", False)]
    abstentions = len(votes) - len(cast)

    if not cast:
        return Verdict(
            tally={convict_word: 0, acquit_word: 0},
            outcome="No verdict (the jury returned no ballots)",
            unanimous=False,
            hung=True,
            dissent_summary="No juror was able to return a ballot.",
            abstentions=abstentions,
        )

    convict = sum(1 for v in cast if v.vote == "convict")
    acquit = len(cast) - convict
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
            v for v in cast
            if (v.vote == "acquit" and outcome == convict_word)
            or (v.vote == "convict" and outcome == acquit_word)
        ]
        names = ", ".join(v.juror_name for v in dissenters)
        dissent_summary = f"Majority verdict. Dissenting ({losing}): {names}."

    if abstentions:
        dissent_summary += (
            f" ({abstentions} juror(s) returned no ballot and were excluded from "
            "the count.)"
        )

    return Verdict(
        tally=tally,
        outcome=outcome,
        unanimous=unanimous,
        hung=hung,
        dissent_summary=dissent_summary,
        abstentions=abstentions,
    )


def _project_defendant_ballots(votes: list[JurorVote], d: Defendant) -> list[JurorVote]:
    """Each juror's ballot FOR ONE accused — their DefendantVote, or the top-level.

    Carries element_findings and per-charge votes through so a downstream per-charge
    tally still works. Names are matched on meaning (`_pick_by_label`), so a juror
    who writes "Marlowe" for "Dana Marlowe" is still counted on that
    accused; only a genuinely absent entry falls back to the top-level vote.
    """
    out: list[JurorVote] = []
    for v in votes:
        src = _pick_by_label(v.defendant_votes, "defendant_name", d.name) or v
        out.append(
            JurorVote(
                juror_name=v.juror_name,
                verdict=src.verdict,
                vote=src.vote,
                confidence=src.confidence,
                reasoning=src.reasoning,
                abstained=v.abstained,
                sample_agreement=v.sample_agreement,
                element_findings=list(getattr(src, "element_findings", [])),
                charge_votes=list(getattr(src, "charge_votes", [])),
            )
        )
    return out


def _projection_gaps(votes: list[JurorVote], defendants: list, charges: list) -> list[str]:
    """Ballots where a named accused or charge had no matching entry.

    These are the cases where the tally silently falls back to a juror's top-level
    vote, so they are worth surfacing rather than hiding: a run with many gaps has a
    less trustworthy per-charge breakdown than one with none.
    """
    gaps: list[str] = []
    multi_d, multi_c = len(defendants) > 1, len(charges) > 1
    for v in votes:
        if v.abstained:
            continue
        if multi_d:
            for d in defendants:
                dv = _pick_by_label(v.defendant_votes, "defendant_name", d.name)
                if dv is None:
                    gaps.append(f"{v.juror_name}: no ballot entry for {d.name}")
                elif multi_c:
                    for c in charges:
                        if _pick_by_label(dv.charge_votes, "charge_label", c.label) is None:
                            gaps.append(f"{v.juror_name}/{d.name}: no entry for {c.label}")
        elif multi_c:
            for c in charges:
                if _pick_by_label(v.charge_votes, "charge_label", c.label) is None:
                    gaps.append(f"{v.juror_name}: no entry for {c.label}")
    return gaps


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
            src = _pick_by_label(v.charge_votes, "charge_label", c.label) or v
            projected.append(
                JurorVote(
                    juror_name=v.juror_name, verdict=src.verdict, vote=src.vote,
                    confidence=src.confidence, reasoning=src.reasoning,
                    abstained=v.abstained,
                )
            )
        base = _tally_votes(projected, case_type)
        out.append(
            ChargeVerdict(
                charge_label=c.label, tally=base.tally, outcome=base.outcome,
                unanimous=base.unanimous, hung=base.hung, dissent_summary=base.dissent_summary,
                abstentions=base.abstentions,
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
        if getattr(v, "abstained", False):
            sig.append((v.juror_name, "abstain"))
        elif v.defendant_votes:
            inner = tuple(
                sorted((dv.defendant_name.strip().lower(), charge_sig(dv)) for dv in v.defendant_votes)
            )
            sig.append((v.juror_name, inner))
        else:
            sig.append((v.juror_name, charge_sig(v)))
    return tuple(sorted(sig, key=lambda x: x[0]))


def _ballot_signature(v: JurorVote) -> tuple:
    """One juror's full position (every accused, every charge) as a hashable key."""
    return _tally_signature([v])[0][1:]


def _modal_ballot(ballots: list[JurorVote]) -> JurorVote:
    """The ballot a juror gave most often across independent samples.

    Self-consistency: sampling one ballot per juror makes the verdict a draw from the
    model's distribution rather than a reading of the evidence, and on a close case
    that draw decides the trial. Taking each juror's MODAL position across K samples
    keeps a whole coherent ballot (rather than stitching a Frankenstein one out of
    per-element majorities) and records how stable that juror actually was. Ties go
    to the most confident of the tied ballots.
    """
    if len(ballots) == 1:
        return ballots[0]
    counts = Counter(_ballot_signature(b) for b in ballots)
    top = max(counts.values())
    winners = [sig for sig, n in counts.items() if n == top]
    pool = [b for b in ballots if _ballot_signature(b) in winners]
    chosen = max(pool, key=lambda b: b.confidence)
    chosen.sample_agreement = round(top / len(ballots), 3)
    return chosen


def _has_conviction(verdict: Verdict, convict_word: str) -> bool:
    """Did anything at all result in a conviction / finding of liability?

    Checked across every axis, because a multi-charge multi-accused verdict can
    convict on one count while the headline outcome reads as an acquittal.
    """
    if verdict.outcome == convict_word:
        return True
    for dv in verdict.per_defendant:
        if dv.outcome == convict_word or any(c.outcome == convict_word for c in dv.per_charge):
            return True
    return any(c.outcome == convict_word for c in verdict.per_charge)


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
        logger.exception("Trial aborted: %s", exc)
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
            logger.warning("%s could not make a statement in %s: %s", speaker, phase, exc)
            text = f"(The {speaker} was unable to make a statement: {exc})"
        log(speaker, text)
        return TrialEvent(phase=phase, speaker=speaker, kind="message", content=text)

    # --- 0. Run manifest -------------------------------------------------
    # Emitted first so any transcript can be traced back to the exact settings that
    # produced it — otherwise two runs that disagree are impossible to compare.
    yield TrialEvent(
        phase="Intake", speaker="System", kind="structured",
        content="Trial configuration.",
        data={
            "_manifest": True,
            **RunManifest(
                model=case.model or get_settings().model,
                case_type=case.case_type,
                jury_size=case.jury_size,
                argument_rounds=case.argument_rounds,
                deliberation_rounds=case.deliberation_rounds,
                deliberation_style=case.deliberation_style,
                verdict_passes=case.verdict_passes,
                proof_threshold=_proof_threshold(case),
                strict_elements=case.strict_elements,
                calibrated_proof=case.calibrated_proof,
                evidence_digest=case.evidence_digest,
                grounding_check=case.grounding_check,
                straw_poll=case.straw_poll,
            ).model_dump(),
        },
    )

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
    if case.charges:
        pinned = "\n".join(
            f"- {c.label}\n" + "\n".join(f"    * {e}" for e in c.elements)
            for c in case.charges
        )
        intake_prompt += (
            "\n\nThe charges and their essential legal elements are SETTLED and given "
            "to you below. Echo them back in 'charges' EXACTLY as written — do not "
            "reword, add, drop, or reorder any element — and mirror the first charge's "
            f"elements in 'elements':\n{pinned}"
        )
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
        except Exception as exc:
            logger.warning("Fact-check on %s (%s) skipped: %s", speaker, phase, exc)
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
    # Charges pinned on the case file WIN: the elements of an offence are settled
    # law, and letting the intake clerk re-derive them each run quietly moves the
    # verdict, because conviction is an AND across whatever list it produced.
    charges = list(case.charges) if case.charges else list(sc.charges)
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
        except Exception as exc:
            logger.warning("No case theory for the %s: %s", side_label, exc)
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
        except Exception as exc:
            logger.warning("Casting failed; using neutral personas: %s", exc)
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
    except Exception as exc:
        logger.warning("Jury pool generation failed; using fallback jurors: %s", exc)
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
                    except Exception as exc:
                        logger.warning("Objection check by %s failed: %s", objector_name, exc)
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
                        except Exception as exc:
                            logger.warning("Objection ruling failed; overruled by default: %s", exc)
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
        except Exception as exc:
            logger.warning("Directed-verdict motion check failed: %s", exc)
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
            except Exception as exc:
                logger.warning("Directed-verdict ruling failed; dismissed by default: %s", exc)
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

    # --- 6b. Neutral evidence digest, element by element -----------------
    # Before the jury retires, a neutral clerk maps the record onto the elements:
    # what supports each one, what undercuts it, what the record simply does not
    # contain. Jurors otherwise have to hold ~150KB of advocacy in their heads and
    # end up weighing whichever closing was more stirring; the digest anchors
    # deliberation to the evidence and is a large part of why two runs of the same
    # case stop disagreeing.
    digest_block = ""
    if case.evidence_digest:
        yield TrialEvent(
            phase="Digest", speaker="System", kind="phase",
            content="The Evidence, Element by Element",
        )
        digest_agent = build_digest_agent(model)
        charge_spec = "\n".join(
            f"  Charge: {c.label}\n"
            + "\n".join(f"    - {e}" for e in c.elements)
            for c in charges
        )
        accused_note = (
            "\n\nThere are MULTIPLE co-accused: "
            + ", ".join(d.name for d in defendants)
            + ". Cover each element for EACH accused, keeping their positions distinct."
            if multi else ""
        )
        try:
            digest = await run_structured(
                digest_agent,
                f"{case_brief}\n\nCHARGES AND ELEMENTS TO MAP:\n{charge_spec}{accused_note}\n\n"
                f"FULL TRIAL RECORD:\n{transcript_text()}\n\n"
                "Compile the evidence digest. Return ONLY an EvidenceDigest JSON.",
                EvidenceDigest, require_english=True,
            )
        except Exception as exc:
            logger.warning("Evidence digest could not be compiled: %s", exc)
            digest = None
        if digest and digest.charges:
            lines = ["=== EVIDENCE DIGEST (neutral; compiled by the court clerk) ==="]
            for ce in digest.charges:
                lines.append(f"Charge — {ce.charge_label}:")
                for ee in ce.elements:
                    lines.append(f"  Element: {ee.element}")
                    for s in ee.supporting:
                        lines.append(f"    [supports] {s}")
                    for u in ee.undermining:
                        lines.append(f"    [undercuts] {u}")
                    for g in ee.gaps:
                        lines.append(f"    [not in the record] {g}")
            if digest.undisputed:
                lines.append("Undisputed: " + "; ".join(digest.undisputed))
            if digest.disputed:
                lines.append("In dispute: " + "; ".join(digest.disputed))
            lines.append(
                "This digest is a neutral aid, not evidence and not a conclusion. Where "
                "it and the transcript differ, the transcript governs."
            )
            digest_block = "\n".join(lines)
            yield TrialEvent(
                phase="Digest", speaker="Court Clerk", kind="structured",
                content="The evidence has been summarised, element by element.",
                data={"_digest": True, **digest.model_dump()},
            )
        else:
            yield TrialEvent(
                phase="Digest", speaker="Court Clerk", kind="message",
                content=(
                    "No digest could be compiled; the jury will work from the "
                    "transcript alone."
                ),
            )

    # --- 7. Jury deliberation -------------------------------------------
    yield TrialEvent(
        phase="Deliberation", speaker="System", kind="phase", content="The Jury Deliberates"
    )
    full_transcript = transcript_text()
    # Everything a juror reasons from: the neutral digest first, then the raw record.
    jury_record = (f"{digest_block}\n\n" if digest_block else "") + (
        f"Full trial transcript:\n{full_transcript}"
    )
    threshold = _proof_threshold(case) if case.calibrated_proof else None
    # The authoritative element list per charge — what a juror's findings are checked
    # against, so partial or invented findings can't decide the verdict.
    primary_elements = list(charges[0].elements) if charges else []

    def _elements_for(label: str) -> list[str]:
        match = _pick_by_label(charges, "label", label)
        return list(match.elements) if match is not None else primary_elements

    def _apply_element_logic(vote: JurorVote) -> None:
        """Re-derive every convict/acquit signal on a ballot from its element findings.

        Applied at all four levels (top, per-charge, per-defendant, per-defendant
        per-charge), each checked against that charge's real element list.
        """
        strict = case.strict_elements

        def derive(obj, required):
            return _derive_vote(
                obj.element_findings, obj.vote, obj.verdict,
                required=required, strict=strict, threshold=threshold,
            )

        vote.vote = derive(vote, primary_elements if not multi_c else None)
        for cv in vote.charge_votes:
            cv.vote = derive(cv, _elements_for(cv.charge_label))
        if vote.charge_votes:
            vote.vote = _headline_from_charges(vote.charge_votes)
        for dv in vote.defendant_votes:
            dv.vote = derive(dv, primary_elements if not multi_c else None)
            for cv in dv.charge_votes:
                cv.vote = derive(cv, _elements_for(cv.charge_label))
            if dv.charge_votes:
                dv.vote = _headline_from_charges(dv.charge_votes)

    async def get_vote(persona: JurorPersona, extra: str = "", passes: int = 1) -> JurorVote:
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
        standard_note = (
            "\n\nFor EVERY element finding also give \"probability\": your honest "
            f"percentage (0-100) that the element is proven. \"{standard}\" means you "
            f"must be at least {threshold}% sure before you may mark an element proven; "
            "below that, the element is NOT proven."
            if threshold is not None else ""
        )
        prompt = (
            f"Your persona — name: {persona.name}; background: {persona.background}; "
            f"disposition: {persona.disposition}.\n\n{case_brief}\n\n"
            f"{jury_record}{per_def}{charge_note}{standard_note}{extra}"
        )

        async def one_ballot() -> JurorVote | None:
            try:
                vote = await run_structured(juror_agent, prompt, JurorVote, require_english=True)
            except Exception as exc:
                logger.warning("No ballot from juror %s: %s", persona.name, exc)
                return None
            vote.juror_name = persona.name  # keep the persona name even if the model drifts
            # The verdict is driven by the per-element findings: convict only if every
            # essential element of that charge is proven to the standard.
            _apply_element_logic(vote)
            return vote

        ballots = [b for b in await asyncio.gather(
            *[one_ballot() for _ in range(max(1, passes))]
        ) if b is not None]
        if not ballots:
            # No ballot could be obtained. ABSTAIN — never invent a vote. A fabricated
            # acquittal here used to be enough, on its own, to hang a criminal jury.
            return JurorVote(
                juror_name=persona.name,
                verdict="(no ballot)",
                vote="acquit",
                confidence=1,
                reasoning="(This juror was unable to return a ballot; excluded from the count.)",
                abstained=True,
            )
        return _modal_ballot(ballots)

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
            f"{case_brief}\n\n{jury_record}\n\n"
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
            f"{jury_record}\n\n"
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

    def _rotate(seq: list, n: int) -> list:
        """Rotate the speaking/reporting order so the same juror never anchors twice.

        Whoever speaks first sets the frame the rest respond to. Fixing that order
        across rounds bakes one juror's reading into every round of the discussion.
        """
        if not seq:
            return seq
        k = n % len(seq)
        return seq[k:] + seq[:k]

    votes: list[JurorVote] = []
    verdict: Verdict | None = None
    prev_sig = None
    exhorted = False
    exhortation_text = ""
    # One extra round is held in reserve for the judge's exhortation after a
    # deadlock, so a jury that stalls gets a genuine second attempt.
    max_rounds = case.deliberation_rounds + (1 if case.deadlock_exhortation else 0)
    d_round = 0
    while d_round < max_rounds:
        d_round += 1
        # EVERY round gets the full sample count, because any round can be the last:
        # a jury that settles unanimously in round one ends the deliberation there,
        # and that ballot decides the trial as surely as a final-round one would.
        passes = case.verdict_passes
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
            for p in _rotate(personas, d_round - 1):
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
                f"- {v.juror_name} voted {v.verdict}: {v.reasoning}"
                for v in _rotate([x for x in votes if not x.abstained], d_round - 1)
            )
            tally_line = ", ".join(f"{k}: {n}" for k, n in verdict.tally.items())
            extra = (
                f"\n\nThis is deliberation round {d_round}. The running tally is "
                f"{tally_line}. Here is what the other jurors said:\n{others}\n\n"
                "Reconsider and give your vote for THIS round. A head-count is not an "
                "argument: change your mind only if a REASON you have heard actually "
                "answers your doubt, and hold firm if none does."
            )
        if exhortation_text:
            extra += f"\n\nThe judge has addressed you:\n{exhortation_text}"

        votes = list(await asyncio.gather(*[get_vote(p, extra, passes) for p in personas]))
        if d_round == 1:
            prefix = ""
        elif d_round >= case.deliberation_rounds:
            prefix = "FINAL: "
        else:
            prefix = "(revised) "
        for v in votes:
            stability = (
                f" · agreement {int(v.sample_agreement * 100)}%"
                if passes > 1 and not v.abstained else ""
            )
            yield TrialEvent(
                phase="Deliberation",
                speaker=f"Juror — {v.juror_name}",
                kind="structured",
                content=f"{prefix}{v.verdict} (confidence {v.confidence}/10){stability}",
                data=v.model_dump(),
            )
        verdict = tally(votes)
        # Exit early only if the verdict is settled AND the carrying side holds it
        # with real conviction — a shaky (low-confidence) consensus gets another
        # (bounded) round instead of ending immediately.
        if (_all_settled(verdict) and _settled_with_conviction(votes, verdict)) or len(personas) <= 1:
            break
        sig = _tally_signature(votes)
        stalled = d_round > 1 and sig == prev_sig
        prev_sig = sig
        if not stalled and d_round >= case.deliberation_rounds:
            break  # rounds exhausted, and the room is still moving — that is the verdict
        if stalled:
            # A divided jury that has stopped moving. Once, the judge sends them back
            # with an exhortation (they are asked to reconsider, never to surrender a
            # doubt they honestly hold); a second stall is a genuine deadlock.
            if not (case.deadlock_exhortation and not exhorted):
                break
            exhorted = True
            yield TrialEvent(
                phase="Deliberation", speaker="System", kind="phase",
                content="The Jury Reports a Deadlock",
            )
            split = _delib_status(verdict).strip() or "The jury is divided."
            ev = await speak(
                judge,
                f"{judge_persona_text}\n\nThe jury has sent word that it is divided and is "
                f"not making progress. {split}\n\nDeliver a short exhortation: ask them to "
                "return and try again with an open mind, to listen to each other's reasons "
                "and re-examine their own — while making it absolutely clear that no juror "
                "should surrender an honestly held view merely to reach a verdict or to "
                "agree with the majority, and that a jury which genuinely cannot agree is "
                "entitled to say so. Do NOT suggest what the verdict should be.\n\n"
                f"{case_brief}",
                phase="Deliberation", speaker="Judge",
            )
            exhortation_text = ev.content
            yield ev
            room.append(f"The Judge (to the jury): {ev.content}")

    # How the room moved from the private straw poll to the final vote.
    if straw_votes:
        yield TrialEvent(
            phase="Deliberation",
            speaker="Court Clerk",
            kind="structured",
            content="How the room moved after deliberation.",
            data={"_movement": True, **_straw_movement(straw_votes, votes, defendants, charges, case.case_type).model_dump()},
        )

    # Ballot health for this verdict: abstentions, unmatched entries, and how stable
    # each juror was across samples. A verdict resting on shaky ballots should not
    # look as authoritative as one resting on solid ones, so we say which it is.
    gaps = _projection_gaps(votes, defendants, charges)
    abstained = [v.juror_name for v in votes if v.abstained]
    agreements = [v.sample_agreement for v in votes if not v.abstained]
    mean_agreement = round(sum(agreements) / len(agreements), 3) if agreements else 1.0
    if gaps or abstained or (case.verdict_passes > 1):
        # `passes` is what the DECIDING round actually sampled. Reporting the config
        # value instead would show a reassuring "100% agreement" for a ballot that
        # was never resampled at all.
        agreement_line = (
            f"; mean self-agreement {int(mean_agreement * 100)}% over {passes} samples"
            if passes > 1 else "; single-sample ballots (not resampled)"
        )
        yield TrialEvent(
            phase="Deliberation", speaker="Court Clerk", kind="structured",
            content=(
                f"Ballot integrity: {len(votes) - len(abstained)}/{len(votes)} ballots "
                f"counted{agreement_line}."
            ),
            data={
                "_diagnostics": True,
                "ballots_counted": len(votes) - len(abstained),
                "ballots_expected": len(votes),
                "abstentions": abstained,
                "mean_sample_agreement": mean_agreement if passes > 1 else None,
                "verdict_passes": passes,
                "deciding_round": d_round,
                "unmatched_entries": gaps,
            },
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
    # Sentencing detail is only ever coherent where something was actually proved.
    # The model will sometimes list aggravating factors and a sentencing range under
    # an acquittal or a mistrial; nothing downstream catches it, so we do it here.
    if not _has_conviction(verdict, convict_word):
        ruling.aggravating_factors = []
        ruling.mitigating_factors = []
        ruling.sentencing_range = ""
        ruling.restitution = ""
        ruling.conditions = []
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

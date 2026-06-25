"""Unit tests for the pure trial logic — no network / API calls.

Run: pytest -q
"""

from __future__ import annotations

from app.llm_utils import contains_cjk, extract_json, parse_json_lenient, strip_think
from app.orchestrator import (
    _all_settled,
    _defendant_roster,
    _normalize_vote,
    _tally_signature,
    _tally_votes,
    _tally_votes_multi,
)
from app.schemas import CaseInput, Defendant, DefendantVote, JurorVote, StructuredCase


def _jurors(convict: int, acquit: int) -> list[JurorVote]:
    return (
        [JurorVote(juror_name=f"C{i}", verdict="guilty", vote="convict") for i in range(convict)]
        + [JurorVote(juror_name=f"A{i}", verdict="not guilty", vote="acquit") for i in range(acquit)]
    )


# ---------------------------------------------------------------------------
# Verdict tally — criminal unanimity vs civil majority
# ---------------------------------------------------------------------------
def test_criminal_requires_unanimity():
    assert _tally_votes(_jurors(12, 0), "criminal").outcome == "Guilty"
    assert _tally_votes(_jurors(0, 12), "criminal").outcome == "Not Guilty"
    assert _tally_votes(_jurors(12, 0), "criminal").unanimous is True

    for split in [(11, 1), (7, 5), (1, 11), (6, 6)]:
        v = _tally_votes(_jurors(*split), "criminal")
        assert v.hung is True, split
        assert "unanim" in v.outcome.lower()


def test_civil_is_majority():
    assert _tally_votes(_jurors(7, 5), "civil").outcome == "Liable"
    assert _tally_votes(_jurors(5, 7), "civil").outcome == "Not Liable"
    tie = _tally_votes(_jurors(6, 6), "civil")
    assert tie.hung is True and "majority" in tie.dissent_summary.lower()


def test_tally_counts_by_vote_enum_not_text():
    # Verdict prose is misleading; the enum is authoritative.
    votes = [JurorVote(juror_name="X", verdict="he is clearly guilty", vote="acquit")] * 1
    v = _tally_votes(votes, "criminal")
    assert v.tally["Not Guilty"] == 1 and v.tally["Guilty"] == 0


def test_dissent_summary_wording():
    crim = _tally_votes(_jurors(7, 5), "criminal")
    assert "unanimous" in crim.dissent_summary.lower()
    civ = _tally_votes(_jurors(8, 4), "civil")
    assert "dissent" in civ.dissent_summary.lower()


# ---------------------------------------------------------------------------
# Robust vote normalization (the old brittle "contains 'not'" bug)
# ---------------------------------------------------------------------------
def test_normalize_vote_handles_cannot():
    # "cannot ... guilty" used to be misread as acquittal because it contains "not".
    assert _normalize_vote("The Crown cannot be doubted, guilty", "acquit") == "convict"


def test_normalize_vote_clear_cases_and_fallback():
    assert _normalize_vote("not guilty", "convict") == "acquit"
    assert _normalize_vote("not liable", "convict") == "acquit"
    assert _normalize_vote("guilty", "acquit") == "convict"
    # Ambiguous text → fall back to the model's explicit enum.
    assert _normalize_vote("undecided", "convict") == "convict"
    assert _normalize_vote("", "acquit") == "acquit"


# ---------------------------------------------------------------------------
# Multi-defendant verdicts
# ---------------------------------------------------------------------------
def _multi_ballot(name: str, picks: dict[str, str]) -> JurorVote:
    return JurorVote(
        juror_name=name,
        defendant_votes=[
            DefendantVote(defendant_name=d, verdict=("guilty" if p == "convict" else "not guilty"), vote=p)
            for d, p in picks.items()
        ],
    )


def test_multi_defendant_separate_verdicts():
    defs = [Defendant(name="Marlowe", role="CEO"), Defendant(name="Vance", role="contractor")]
    votes = [_multi_ballot(f"J{i}", {"Marlowe": "convict", "Vance": "acquit"}) for i in range(3)]
    v = _tally_votes_multi(votes, defs, "criminal")
    out = {d.defendant_name: d.outcome for d in v.per_defendant}
    assert out["Marlowe"] == "Guilty"
    assert out["Vance"] == "Not Guilty"
    # Top-level mirrors the first defendant for backward-compat consumers.
    assert v.outcome == v.per_defendant[0].outcome


def test_multi_missing_entry_falls_back_to_toplevel():
    defs = [Defendant(name="Marlowe"), Defendant(name="Vance")]
    # Juror only votes on Marlowe; top-level vote covers the gap for Vance.
    v = JurorVote(juror_name="J", verdict="not guilty", vote="acquit",
                  defendant_votes=[DefendantVote(defendant_name="Marlowe", verdict="guilty", vote="convict")])
    res = _tally_votes_multi([v], defs, "criminal")
    out = {d.defendant_name: d.outcome for d in res.per_defendant}
    assert out["Marlowe"] == "Guilty"
    assert out["Vance"] == "Not Guilty"  # fell back to the acquit top-level


# ---------------------------------------------------------------------------
# Deliberation helpers
# ---------------------------------------------------------------------------
def test_tally_signature_and_all_settled():
    a = _jurors(2, 1)
    assert _tally_signature(a) == _tally_signature(_jurors(2, 1))
    assert _tally_signature(a) != _tally_signature(_jurors(1, 2))
    assert _all_settled(_tally_votes(_jurors(3, 0), "criminal")) is True
    assert _all_settled(_tally_votes(_jurors(2, 1), "criminal")) is False


def test_defendant_roster_fallback_and_cap():
    sc = StructuredCase(case_caption="R. v. Smith")
    case = CaseInput(title="t", charge_or_claim="c", your_side="s")
    roster = _defendant_roster(case, sc)
    assert len(roster) == 1 and roster[0].name == "Smith"

    many = CaseInput(
        title="t", charge_or_claim="c", your_side="s",
        defendants=[Defendant(name=f"D{i}") for i in range(6)],
    )
    assert len(_defendant_roster(many, sc)) == 4  # hard cap


# ---------------------------------------------------------------------------
# LLM utils
# ---------------------------------------------------------------------------
def test_contains_cjk():
    assert contains_cjk("hello world") is False
    assert contains_cjk("the verdict is 无罪") is True


def test_strip_think():
    assert strip_think("<think>hidden</think>answer") == "answer"
    assert strip_think("before<think>unclosed to end") == "before"


def test_extract_json_fenced_and_raw():
    assert parse_json_lenient(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert parse_json_lenient(extract_json('noise {"b": 2} trailing')) == {"b": 2}


def test_parse_json_lenient_tolerates_control_chars():
    # A literal newline inside a string value would break strict json.loads.
    assert parse_json_lenient('{"x": "line1\nline2"}')["x"] == "line1\nline2"

"""Unit tests for the pure trial logic — no network / API calls.

Run: pytest -q
"""

from __future__ import annotations

from app.llm_utils import contains_cjk, extract_json, parse_json_lenient, strip_think
from app.orchestrator import (
    _all_settled,
    _charge_directives,
    _defendant_roster,
    _derive_vote,
    _directed_acquittal,
    _exam_kinds,
    _fallback_cast,
    _merge_reports,
    _normalize_vote,
    _persona_text,
    _record_block,
    _settled_with_conviction,
    _straw_movement,
    _tally,
    _tally_signature,
    _tally_votes,
    _tally_votes_multi,
)
from app.schemas import (
    AgreedRecord,
    CaseInput,
    CaseStrategy,
    Charge,
    ChargeVote,
    Defendant,
    DefendantVote,
    DirectedVerdictMotion,
    ElementFinding,
    GroundingFlag,
    GroundingReport,
    JudgeRuling,
    JurorVote,
    RolePersona,
    StructuredCase,
    TrialCast,
    Witness,
)


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
# Element-driven verdicts (the judge -> jury "via legal elements" wiring)
# ---------------------------------------------------------------------------
def _findings(*proven: bool) -> list[ElementFinding]:
    return [ElementFinding(element=f"element {i}", proven=p) for i, p in enumerate(proven)]


def test_derive_vote_convicts_only_when_every_element_proven():
    assert _derive_vote(_findings(True, True, True), "acquit", "guilty") == "convict"
    assert _derive_vote(_findings(True, False, True), "convict", "guilty") == "acquit"
    assert _derive_vote(_findings(False), "convict", "guilty") == "acquit"


def test_derive_vote_one_unproven_element_acquits_over_model_vote():
    # Even if the model says "convict" / "guilty", a single unproven essential
    # element forces an acquittal — the verdict turns on the law, not the gut.
    assert _derive_vote(_findings(True, True, False, True), "convict", "guilty") == "acquit"


def test_derive_vote_falls_back_to_text_when_no_findings():
    # No element findings -> behave exactly like _normalize_vote.
    assert _derive_vote([], "convict", "guilty") == "convict"
    assert _derive_vote([], "convict", "not guilty") == "acquit"
    assert _derive_vote([], "acquit", "") == "acquit"


def test_caseinput_deliberation_style_defaults_to_dialogue():
    c = CaseInput(title="t", charge_or_claim="c", your_side="s")
    assert c.deliberation_style == "dialogue"
    c2 = CaseInput(title="t", charge_or_claim="c", your_side="s", deliberation_style="poll")
    assert c2.deliberation_style == "poll"


def test_case_strategy_parses_and_defaults_are_forgiving():
    s = CaseStrategy.model_validate({
        "theory": "A failed business, not a fraud.",
        "strongest_points": ["no skimming", "returns paid for years"],
        "opponents_best_point": "The broker held no investor accounts.",
        "rebuttal": "Client money never sat there, so that proves nothing.",
    })
    assert s.theory and len(s.strongest_points) == 2 and s.opponents_best_point
    # A missing key must never crash a trial — every field defaults.
    assert CaseStrategy().strongest_points == []


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


# ---------------------------------------------------------------------------
# Agreed Record (anti-hallucination ledger)
# ---------------------------------------------------------------------------
def test_record_block_empty_is_blank():
    assert _record_block(AgreedRecord()) == ""


def test_record_block_includes_figures_and_rules():
    rec = AgreedRecord(figures=["amount raised: $250,000"], parties=["Dana Marlowe"])
    block = _record_block(rec)
    assert "amount raised: $250,000" in block
    assert "single source of truth" in block.lower()
    assert "GROUNDING RULES" in block


def test_structured_case_defaults_agreed_record():
    assert StructuredCase().agreed_record.parties == []


# ---------------------------------------------------------------------------
# Confidence-aware early-exit
# ---------------------------------------------------------------------------
def _conf_jurors(specs: list[tuple[str, int]]) -> list[JurorVote]:
    return [
        JurorVote(juror_name=f"J{i}", vote=v, verdict=("guilty" if v == "convict" else "not guilty"),
                  confidence=c)
        for i, (v, c) in enumerate(specs)
    ]


def test_settled_with_conviction_blocks_low_confidence_exit():
    votes = _conf_jurors([("convict", 2), ("convict", 2), ("convict", 3)])
    v = _tally_votes(votes, "criminal")
    assert _settled_with_conviction(votes, v) is False  # shaky → keep deliberating


def test_settled_with_conviction_allows_confident_exit():
    votes = _conf_jurors([("convict", 8), ("convict", 9), ("convict", 7)])
    v = _tally_votes(votes, "criminal")
    assert _settled_with_conviction(votes, v) is True


def test_settled_with_conviction_ignores_minority_confidence():
    # Carrying side (acquit) is confident; a lone low-confidence dissenter is irrelevant.
    votes = _conf_jurors([("acquit", 9), ("acquit", 8), ("acquit", 8), ("convict", 1)])
    v = _tally_votes(votes, "civil")
    assert _settled_with_conviction(votes, v) is True


# ---------------------------------------------------------------------------
# Jury-charge directives (W.(D.), circumstantial, s.21, inferences)
# ---------------------------------------------------------------------------
def test_charge_directives_includes_wd_for_criminal():
    case = CaseInput(title="t", charge_or_claim="c", your_side="s", case_type="criminal")
    text = " ".join(_charge_directives(case, [Defendant(name="A")]))
    assert "W.(D.)" in text and "circumstantial" in text.lower() and "inference" in text.lower()


def test_charge_directives_party_liability_only_multi_defendant():
    case = CaseInput(title="t", charge_or_claim="c", your_side="s", case_type="criminal")
    one = " ".join(_charge_directives(case, [Defendant(name="A")]))
    two = " ".join(_charge_directives(case, [Defendant(name="A"), Defendant(name="B")]))
    assert "s.21" not in one
    assert "s.21" in two


def test_charge_directives_civil_minimal():
    case = CaseInput(title="t", charge_or_claim="c", your_side="s", case_type="civil")
    ds = _charge_directives(case, [Defendant(name="A")])
    assert len(ds) == 1 and "inference" in ds[0].lower()  # only the inference instruction


# ---------------------------------------------------------------------------
# Structured sentencing fields on JudgeRuling
# ---------------------------------------------------------------------------
def test_judge_ruling_new_fields_default_empty():
    r = JudgeRuling()
    assert r.aggravating_factors == [] and r.mitigating_factors == [] and r.sentencing_range == ""
    assert r.conditions == [] and r.restitution == ""


def test_judge_ruling_parses_full_sentencing():
    r = JudgeRuling.model_validate({
        "verdict_acknowledgement": "ack", "reasoning": "r", "sentence_or_remedy": "3 years",
        "aggravating_factors": ["breach of trust"], "mitigating_factors": ["no record"],
        "sentencing_range": "2-5 years", "restitution": "$1M", "conditions": ["no contact"],
    })
    assert r.sentencing_range == "2-5 years" and r.aggravating_factors == ["breach of trust"]


# ---------------------------------------------------------------------------
# Straw-poll movement (anti-herding)
# ---------------------------------------------------------------------------
def test_straw_movement_detects_flips():
    defs, chgs = [Defendant(name="X")], [Charge(label="c")]
    initial = [JurorVote(juror_name="A", vote="convict", verdict="guilty"),
               JurorVote(juror_name="B", vote="acquit", verdict="not guilty")]
    final = [JurorVote(juror_name="A", vote="acquit", verdict="not guilty"),
             JurorVote(juror_name="B", vote="acquit", verdict="not guilty")]
    mv = _straw_movement(initial, final, defs, chgs, "criminal")
    assert mv.flips == ["A: convict → acquit"]
    assert mv.initial_tally["Guilty"] == 1 and mv.final_tally["Not Guilty"] == 2


def test_straw_movement_no_change():
    defs, chgs = [Defendant(name="X")], [Charge(label="c")]
    votes = [JurorVote(juror_name="A", vote="acquit", verdict="not guilty")]
    mv = _straw_movement(votes, votes, defs, chgs, "criminal")
    assert mv.flips == []


# ---------------------------------------------------------------------------
# Grounding / fact-check merge (anti-hallucination)
# ---------------------------------------------------------------------------
def test_merge_reports_unions_and_takes_max_severity():
    r1 = GroundingReport(flags=[GroundingFlag(claim="X is $5M", severity="minor")])
    r2 = GroundingReport(flags=[GroundingFlag(claim="X is $5M", severity="severe")])
    merged = _merge_reports([r1, r2])
    assert len(merged.flags) == 1 and merged.flags[0].severity == "severe"
    assert merged.grounded is False


def test_merge_reports_dedups_distinct_claims():
    r1 = GroundingReport(flags=[GroundingFlag(claim="A")])
    r2 = GroundingReport(flags=[GroundingFlag(claim="B")])
    merged = _merge_reports([r1, r2, None])  # None checker is skipped
    assert {f.claim for f in merged.flags} == {"A", "B"}


def test_merge_reports_empty_is_grounded():
    assert _merge_reports([GroundingReport(), None]).grounded is True


def test_grounding_report_defaults_grounded_true():
    r = GroundingReport()
    assert r.grounded is True and r.flags == []


def test_grounding_flag_severity_enum():
    import pytest
    with pytest.raises(Exception):
        GroundingFlag.model_validate({"claim": "x", "severity": "catastrophic"})


# ---------------------------------------------------------------------------
# Re-direct + directed-verdict motion (procedural realism)
# ---------------------------------------------------------------------------
def test_caseinput_redirect_defaults():
    c = CaseInput(title="t", charge_or_claim="c", your_side="s")
    assert c.redirect is True and c.qa_redirect == 1 and c.allow_directed_verdict is True


def test_exam_kinds_with_and_without_redirect():
    base = CaseInput(title="t", charge_or_claim="c", your_side="s")
    assert _exam_kinds(base) == ["DIRECT", "CROSS", "REDIRECT"]
    off = CaseInput(title="t", charge_or_claim="c", your_side="s", redirect=False)
    assert _exam_kinds(off) == ["DIRECT", "CROSS"]
    zero = CaseInput(title="t", charge_or_claim="c", your_side="s", qa_redirect=0)
    assert _exam_kinds(zero) == ["DIRECT", "CROSS"]


def test_directed_acquittal_all_defendants_acquit():
    v = _directed_acquittal([Defendant(name="A"), Defendant(name="B")], "criminal")
    outs = {d.defendant_name: d.outcome for d in v.per_defendant}
    assert outs == {"A": "Not Guilty", "B": "Not Guilty"}
    assert all(d.unanimous and not d.hung for d in v.per_defendant)


def test_directed_acquittal_respects_case_type():
    v = _directed_acquittal([Defendant(name="A")], "civil")
    assert v.outcome == "Not Liable" and v.hung is False


def test_directed_verdict_motion_defaults_move_false():
    assert DirectedVerdictMotion().move is False


# ---------------------------------------------------------------------------
# Per-charge verdicts
# ---------------------------------------------------------------------------
def _charge_ballot(name: str, picks: dict[str, str]) -> JurorVote:
    return JurorVote(juror_name=name, charge_votes=[
        ChargeVote(charge_label=c, verdict=("guilty" if p == "convict" else "not guilty"), vote=p)
        for c, p in picks.items()
    ])


def test_per_charge_separate_outcomes():
    charges = [Charge(label="Fraud"), Charge(label="Possession")]
    votes = [_charge_ballot(f"J{i}", {"Fraud": "convict", "Possession": "acquit"}) for i in range(3)]
    v = _tally(votes, [Defendant(name="X")], charges, "criminal")
    out = {cv.charge_label: cv.outcome for cv in v.per_charge}
    assert out["Fraud"] == "Guilty" and out["Possession"] == "Not Guilty"


def test_per_charge_missing_entry_falls_back_to_toplevel():
    charges = [Charge(label="Fraud"), Charge(label="Possession")]
    votes = [JurorVote(juror_name="J", vote="acquit", verdict="not guilty",
                       charge_votes=[ChargeVote(charge_label="Fraud", vote="convict", verdict="guilty")])]
    v = _tally(votes, [Defendant(name="X")], charges, "criminal")
    out = {cv.charge_label: cv.outcome for cv in v.per_charge}
    assert out["Fraud"] == "Guilty"          # from the charge vote
    assert out["Possession"] == "Not Guilty"  # fell back to the top-level acquit


def test_per_charge_and_per_defendant_compose():
    defs = [Defendant(name="A"), Defendant(name="B")]
    charges = [Charge(label="Fraud"), Charge(label="Possession")]

    def ballot(n):
        return JurorVote(juror_name=n, defendant_votes=[
            DefendantVote(defendant_name="A", charge_votes=[
                ChargeVote(charge_label="Fraud", vote="convict", verdict="guilty"),
                ChargeVote(charge_label="Possession", vote="acquit", verdict="not guilty")]),
            DefendantVote(defendant_name="B", charge_votes=[
                ChargeVote(charge_label="Fraud", vote="acquit", verdict="not guilty"),
                ChargeVote(charge_label="Possession", vote="acquit", verdict="not guilty")]),
        ])

    v = _tally([ballot(f"J{i}") for i in range(3)], defs, charges, "criminal")
    by = {d.defendant_name: {cv.charge_label: cv.outcome for cv in d.per_charge} for d in v.per_defendant}
    assert by["A"] == {"Fraud": "Guilty", "Possession": "Not Guilty"}
    assert by["B"] == {"Fraud": "Not Guilty", "Possession": "Not Guilty"}


def test_single_charge_emits_no_per_charge():
    v = _tally([JurorVote(juror_name="J", vote="acquit", verdict="not guilty")],
               [Defendant(name="X")], [Charge(label="only")], "criminal")
    assert v.per_charge == []


def test_all_settled_charge_aware():
    charges = [Charge(label="Fraud"), Charge(label="Possession")]
    votes = [
        _charge_ballot("J0", {"Fraud": "convict", "Possession": "acquit"}),
        _charge_ballot("J1", {"Fraud": "convict", "Possession": "acquit"}),
        _charge_ballot("J2", {"Fraud": "acquit", "Possession": "acquit"}),  # Fraud now split
    ]
    v = _tally(votes, [Defendant(name="X")], charges, "criminal")
    assert _all_settled(v) is False  # unanimous on Possession, hung on Fraud


def test_tally_signature_charge_aware():
    a = [_charge_ballot("J", {"Fraud": "convict", "Possession": "acquit"})]
    b = [_charge_ballot("J", {"Fraud": "acquit", "Possession": "acquit"})]
    assert _tally_signature(a) != _tally_signature(b)


# ---------------------------------------------------------------------------
# Auto-cast personalities (counsel / bench / witness)
# ---------------------------------------------------------------------------
def test_persona_text_empty_is_blank():
    assert _persona_text(RolePersona(), "the Crown") == ""


def test_persona_text_includes_name_and_style():
    t = _persona_text(
        RolePersona(name="K. Bayly", background="a veteran prosecutor", style="methodical, understated"),
        "the Crown",
    )
    assert "K. Bayly" in t and "methodical, understated" in t and "the Crown" in t
    assert "do not announce" in t.lower()  # told to embody, not narrate


def test_role_persona_and_trial_cast_defaults():
    assert RolePersona().name == "" and RolePersona().style == ""
    c = TrialCast()
    assert c.crown.name == "" and c.witnesses == []


def test_caseinput_personas_defaults_true():
    assert CaseInput(title="t", charge_or_claim="c", your_side="s").personas is True


def test_fallback_cast_covers_roles_and_witnesses():
    ws = [Witness(name="Priya Nair", role="complainant"), Witness(name="Sam Whitfield", role="expert")]
    cast = _fallback_cast(ws)
    assert cast.crown.style and cast.defense.style and cast.judge.style
    assert [w.name for w in cast.witnesses] == ["Priya Nair", "Sam Whitfield"]
    assert all(w.style for w in cast.witnesses)


def test_caseinput_accepts_pinned_personas():
    c = CaseInput(
        title="t", charge_or_claim="c", your_side="s",
        crown_persona=RolePersona(name="K. Bayly", style="aggressive, theatrical"),
        defense_persona=RolePersona(name="Sam Doe", style="folksy, plain-spoken"),
    )
    assert c.crown_persona.style == "aggressive, theatrical" and c.defense_persona.name == "Sam Doe"
    assert c.judge_persona is None  # unpinned roles stay None and get auto-cast
    # A pinned persona renders to non-empty threading text.
    assert "K. Bayly" in _persona_text(c.crown_persona, "the Crown")

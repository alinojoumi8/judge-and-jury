"""Unit tests for the pure trial logic — no network / API calls.

Run: pytest -q
"""

from __future__ import annotations

from app.llm_utils import contains_cjk, extract_json, parse_json_lenient, strip_think
from app.orchestrator import (
    _align_findings,
    _all_settled,
    _charge_directives,
    _defendant_roster,
    _derive_vote,
    _directed_acquittal,
    _element_proven,
    _exam_kinds,
    _fallback_cast,
    _has_conviction,
    _merge_reports,
    _modal_ballot,
    _normalize_vote,
    _persona_text,
    _pick_by_label,
    _projection_gaps,
    _proof_threshold,
    _record_block,
    _settled_with_conviction,
    _similarity,
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


# ---------------------------------------------------------------------------
# Abstention: a juror whose ballot never arrived must not vote by default
# ---------------------------------------------------------------------------
def test_abstention_is_excluded_not_counted_as_acquittal():
    # 3 convict + 1 juror the model failed on. Counting that failure as an acquittal
    # (the old behaviour) would hang a criminal jury on nothing at all.
    votes = _jurors(3, 0) + [
        JurorVote(juror_name="Broken", verdict="(no ballot)", vote="acquit", abstained=True)
    ]
    v = _tally_votes(votes, "criminal")
    assert v.outcome == "Guilty"
    assert v.hung is False and v.unanimous is True
    assert v.abstentions == 1
    assert v.tally == {"Guilty": 3, "Not Guilty": 0}
    assert "excluded" in v.dissent_summary


def test_all_abstained_yields_no_verdict():
    votes = [JurorVote(juror_name=f"J{i}", abstained=True) for i in range(3)]
    v = _tally_votes(votes, "criminal")
    assert v.hung is True and v.abstentions == 3
    assert "no ballot" in v.outcome.lower() or "no verdict" in v.outcome.lower()


def test_abstention_carries_through_defendant_projection():
    defs = [Defendant(name="A"), Defendant(name="B")]
    votes = _jurors(2, 0) + [JurorVote(juror_name="Broken", abstained=True)]
    v = _tally_votes_multi(votes, defs, "criminal")
    assert all(d.abstentions == 1 and d.outcome == "Guilty" for d in v.per_defendant)


def test_abstention_registers_in_the_tally_signature():
    # An abstention differs from an acquittal, so a juror who recovers on the next
    # round must read as movement rather than as a stalled room.
    out = [JurorVote(juror_name="J", abstained=True)]
    back = [JurorVote(juror_name="J", vote="acquit", verdict="not guilty")]
    assert _tally_signature(out) != _tally_signature(back)


# ---------------------------------------------------------------------------
# Label matching: paraphrased names/charges must still find their entry
# ---------------------------------------------------------------------------
def test_similarity_matches_paraphrase_but_not_unrelated():
    assert _similarity("Marlowe", "Dana Marlowe") > 0.45
    assert _similarity("Fraud over $5,000", "Fraud over $5,000 (s.380(1)(a))") > 0.45
    assert _similarity("Fraud over $5,000", "Possession of proceeds of crime") < 0.45


def test_pick_by_label_tolerates_short_forms():
    entries = [DefendantVote(defendant_name="Dana Marlowe", vote="convict")]
    assert _pick_by_label(entries, "defendant_name", "Marlowe") is entries[0]
    assert _pick_by_label(entries, "defendant_name", "Sam Vance") is None


def test_defendant_projection_matches_shortened_name():
    # The juror wrote "Marlowe"; the roster says "Dana Marlowe". Exact
    # matching silently fell back to the top-level acquit and lost the conviction.
    defs = [Defendant(name="Dana Marlowe")]
    votes = [JurorVote(
        juror_name="J", verdict="not guilty", vote="acquit",
        defendant_votes=[DefendantVote(defendant_name="Marlowe", verdict="guilty", vote="convict")],
    )]
    assert _tally_votes_multi(votes, defs, "criminal").per_defendant[0].outcome == "Guilty"


def test_charge_projection_matches_label_with_extra_citation():
    charges = [Charge(label="Fraud over $5,000 (s.380(1)(a))")]
    votes = [JurorVote(
        juror_name="J", verdict="not guilty", vote="acquit",
        charge_votes=[ChargeVote(charge_label="Fraud over $5,000", verdict="guilty", vote="convict")],
    )]
    v = _tally(votes, [Defendant(name="X")], charges + [Charge(label="Possession")], "criminal")
    out = {cv.charge_label: cv.outcome for cv in v.per_charge}
    assert out["Fraud over $5,000 (s.380(1)(a))"] == "Guilty"
    assert out["Possession"] == "Not Guilty"  # genuinely absent -> top-level fallback


def test_projection_gaps_report_only_genuine_misses():
    defs = [Defendant(name="Alpha Adams"), Defendant(name="Beta Brooks")]
    charges = [Charge(label="Fraud"), Charge(label="Theft")]
    v = JurorVote(juror_name="J", defendant_votes=[
        DefendantVote(defendant_name="Adams", charge_votes=[
            ChargeVote(charge_label="Fraud"), ChargeVote(charge_label="Theft")])
    ])
    gaps = _projection_gaps([v], defs, charges)
    assert len(gaps) == 1 and "Beta Brooks" in gaps[0]  # "Adams" matched; Brooks absent


def test_projection_gaps_ignore_abstentions():
    defs = [Defendant(name="A"), Defendant(name="B")]
    v = JurorVote(juror_name="J", abstained=True)
    assert _projection_gaps([v], defs, [Charge(label="only")]) == []


# ---------------------------------------------------------------------------
# Element coverage: partial or invented findings must not decide the verdict
# ---------------------------------------------------------------------------
def test_align_findings_maps_reworded_elements():
    required = [
        "the accused's subjective knowledge that the act was dishonest",
        "deprivation caused by the dishonest act",
    ]
    findings = [
        ElementFinding(element="Deprivation (loss or risk of loss) caused to the victim", proven=True),
        ElementFinding(element="The accused's subjective knowledge the act was dishonest", proven=False),
    ]
    aligned = _align_findings(required, findings)
    assert aligned[0] is findings[1] and aligned[1] is findings[0]


def test_align_findings_positional_fallback_when_nothing_matches():
    required = ["alpha one", "beta two"]
    findings = [ElementFinding(element="", proven=True), ElementFinding(element="", proven=False)]
    assert _align_findings(required, findings) == findings


def test_align_findings_fills_holes_from_leftovers_in_order():
    # One element matched by wording, one reworded past recognition. The leftover
    # exactly fills the hole, so the juror did address it — reading that as
    # "unaddressed" would acquit on a wording quirk.
    required = ["deprivation caused to the victim", "subjective knowledge of dishonesty"]
    matched = ElementFinding(element="deprivation caused to the victim", proven=True)
    reworded = ElementFinding(element="he knew perfectly well what he was doing", proven=True)
    aligned = _align_findings(required, [matched, reworded])
    assert aligned == [matched, reworded]
    assert _derive_vote([matched, reworded], "convict", "guilty", required=required) == "convict"


def test_align_findings_still_reports_a_genuinely_missing_element():
    required = ["dishonest act", "deprivation", "subjective knowledge"]
    findings = [ElementFinding(element="dishonest act", proven=True)]
    aligned = _align_findings(required, findings)
    assert aligned[0] is findings[0]
    assert aligned[1] is None and aligned[2] is None  # two leftovers short — real gaps


def test_unaddressed_element_acquits_under_strict_coverage():
    required = ["dishonest act", "deprivation", "subjective knowledge of dishonesty"]
    findings = [
        ElementFinding(element="dishonest act", proven=True),
        ElementFinding(element="deprivation", proven=True),
    ]  # never addressed knowledge — the burden is the Crown's, so that is not proven
    assert _derive_vote(findings, "convict", "guilty", required=required, strict=True) == "acquit"
    assert _derive_vote(findings, "convict", "guilty", required=required, strict=False) == "convict"


def test_invented_extra_element_does_not_acquit():
    required = ["dishonest act", "deprivation"]
    findings = [
        ElementFinding(element="dishonest act", proven=True),
        ElementFinding(element="deprivation", proven=True),
        ElementFinding(element="the accused acted alone", proven=False),  # not an element
    ]
    assert _derive_vote(findings, "convict", "guilty", required=required) == "convict"


def test_derive_vote_without_required_keeps_old_behaviour():
    findings = [ElementFinding(element="a", proven=True), ElementFinding(element="b", proven=False)]
    assert _derive_vote(findings, "convict", "guilty") == "acquit"


# ---------------------------------------------------------------------------
# Calibrated standard of proof
# ---------------------------------------------------------------------------
def test_proof_threshold_defaults_by_case_type():
    crim = CaseInput(title="t", charge_or_claim="c", your_side="s", case_type="criminal")
    civ = CaseInput(title="t", charge_or_claim="c", your_side="s", case_type="civil")
    assert _proof_threshold(crim) == 90 and _proof_threshold(civ) == 51
    pinned = CaseInput(title="t", charge_or_claim="c", your_side="s", proof_threshold=75)
    assert _proof_threshold(pinned) == 75


def test_element_proven_downgrades_below_threshold_only():
    high = ElementFinding(element="e", proven=True, probability=95)
    low = ElementFinding(element="e", proven=True, probability=60)
    unsure = ElementFinding(element="e", proven=False, probability=99)
    assert _element_proven(high, 90) is True
    assert _element_proven(low, 90) is False       # own number fails the standard
    assert _element_proven(low, None) is True      # calibration off -> boolean stands
    assert _element_proven(unsure, 90) is False    # never upgrades a "not proven"


def test_missing_probability_falls_back_to_the_boolean():
    f = ElementFinding(element="e", proven=True)
    assert _element_proven(f, 90) is True


def test_calibration_flips_a_verdict_on_a_shaky_element():
    required = ["dishonest act", "subjective knowledge"]
    findings = [
        ElementFinding(element="dishonest act", proven=True, probability=98),
        ElementFinding(element="subjective knowledge", proven=True, probability=65),
    ]
    assert _derive_vote(findings, "convict", "guilty", required=required, threshold=None) == "convict"
    assert _derive_vote(findings, "convict", "guilty", required=required, threshold=90) == "acquit"
    # A civil case takes the same ballot on the balance of probabilities.
    assert _derive_vote(findings, "convict", "liable", required=required, threshold=51) == "convict"


# ---------------------------------------------------------------------------
# Self-consistency: the modal ballot across independent samples
# ---------------------------------------------------------------------------
def _ballot(vote: str, conf: int = 5) -> JurorVote:
    return JurorVote(
        juror_name="J", vote=vote, confidence=conf,
        verdict=("guilty" if vote == "convict" else "not guilty"),
    )


def test_modal_ballot_takes_the_majority_position():
    chosen = _modal_ballot([_ballot("acquit"), _ballot("convict"), _ballot("acquit")])
    assert chosen.vote == "acquit"
    assert chosen.sample_agreement == round(2 / 3, 3)


def test_modal_ballot_unanimous_samples_report_full_agreement():
    chosen = _modal_ballot([_ballot("convict"), _ballot("convict")])
    assert chosen.vote == "convict" and chosen.sample_agreement == 1.0


def test_modal_ballot_tie_breaks_on_confidence():
    chosen = _modal_ballot([_ballot("acquit", 3), _ballot("convict", 9)])
    assert chosen.vote == "convict"


def test_modal_ballot_single_sample_is_untouched():
    only = _ballot("acquit")
    assert _modal_ballot([only]) is only and only.sample_agreement == 1.0


def test_modal_ballot_is_charge_aware():
    # Two samples split on Possession only; the modal ballot must keep both charges.
    def b(possession: str) -> JurorVote:
        return JurorVote(juror_name="J", charge_votes=[
            ChargeVote(charge_label="Fraud", vote="convict"),
            ChargeVote(charge_label="Possession", vote=possession),
        ])
    chosen = _modal_ballot([b("acquit"), b("acquit"), b("convict")])
    picks = {cv.charge_label: cv.vote for cv in chosen.charge_votes}
    assert picks == {"Fraud": "convict", "Possession": "acquit"}


# ---------------------------------------------------------------------------
# Ruling guard: no sentencing where nothing was proved
# ---------------------------------------------------------------------------
def test_has_conviction_across_every_axis():
    acquitted = _tally_votes(_jurors(0, 3), "criminal")
    assert _has_conviction(acquitted, "Guilty") is False
    convicted = _tally_votes(_jurors(3, 0), "criminal")
    assert _has_conviction(convicted, "Guilty") is True


def test_has_conviction_finds_a_lone_guilty_count():
    charges = [Charge(label="Fraud"), Charge(label="Possession")]
    votes = [_charge_ballot(f"J{i}", {"Fraud": "convict", "Possession": "acquit"}) for i in range(3)]
    v = _tally(votes, [Defendant(name="X")], charges, "criminal")
    # Headline outcome mirrors the whole ballot, but one count did convict.
    assert _has_conviction(v, "Guilty") is True


def test_has_conviction_false_when_every_count_acquits():
    charges = [Charge(label="Fraud"), Charge(label="Possession")]
    votes = [_charge_ballot(f"J{i}", {"Fraud": "acquit", "Possession": "acquit"}) for i in range(3)]
    v = _tally(votes, [Defendant(name="X")], charges, "criminal")
    assert _has_conviction(v, "Guilty") is False


# ---------------------------------------------------------------------------
# Reliability settings on CaseInput
# ---------------------------------------------------------------------------
def test_compare_trials_extracts_verdicts_from_a_transcript(tmp_path):
    # The comparison tool is how "is this more reliable?" gets answered, so its
    # parser needs to survive the transcript format it reads.
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("ct", root / "scripts" / "compare_trials.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)

    t = tmp_path / "trial_x.md"
    t.write_text(
        "# Trial transcript\n"
        "**System:** Trial configuration.\n"
        "- Model: `MiniMax-M3` · jury 12\n"
        "## The Jury Deliberates\n"
        "- Ballots counted: 11/12 (deciding round 2)\n"
        "- Mean juror self-agreement: 86% over 3 samples\n"
        "- **this line is outside the verdict section: Guilty**\n"
        "## The Verdict\n"
        "- **Ann Adams — Fraud over $5,000 (s.380(1)(a)): Not Guilty**\n"
        "- **Ann Adams — Possession of proceeds of crime (s.354(1)): Guilty**\n"
        "## The Judge's Ruling\n"
        "- **should be ignored too: Hung jury**\n",
        encoding="utf-8",
    )
    got = ct._parse(t)
    assert got["model"] == "MiniMax-M3"
    assert got["ballots"] == "11/12" and got["agreement"] == "86%"
    assert got["verdicts"] == {
        "Ann Adams — Fraud over $5,000 (s.380(1)(a))": "Not Guilty",
        "Ann Adams — Possession of proceeds of crime (s.354(1))": "Guilty",
    }


def _compare_module():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("ct_q", root / "scripts" / "compare_trials.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_compare_matches_the_same_question_across_wording_drift():
    # The intake clerk names the same count differently run to run. Treating those
    # as different questions reports two identical verdicts as total disagreement.
    ct = _compare_module()
    a = "Sam Vance — Fraud over $5,000 (s.380(1)(a)) — party-liability basis (s.21)"
    b = "Sam Vance — Fraud over $5,000 (s.380(1)(a)), via party liability under s.21"
    assert ct._same_question(a, b) is True


def test_compare_keeps_different_accused_and_charges_apart():
    ct = _compare_module()
    marlowe_fraud = "Dana Marlowe — Fraud over $5,000 (s.380(1)(a)) — party-liability basis (s.21)"
    vance_fraud = "Sam Vance — Fraud over $5,000 (s.380(1)(a)) — party-liability basis (s.21)"
    vance_poss = "Sam Vance — Possession of proceeds of crime (s.354(1)) — party-liability basis (s.21)"
    # Same charge, different accused.
    assert ct._same_question(marlowe_fraud, vance_fraud) is False
    # Same accused, different charge — they share most of their boilerplate wording,
    # so the statute number has to be what tells them apart.
    assert ct._same_question(vance_fraud, vance_poss) is False


def test_compare_matches_a_short_form_of_the_accused():
    ct = _compare_module()
    assert ct._same_question(
        "Dana Marlowe — Fraud over $5,000 (s.380(1)(a))",
        "Marlowe — Fraud over $5,000 (s.380(1)(a))",
    ) is True


def test_compare_split_question_uses_the_first_separator_only():
    ct = _compare_module()
    who, what = ct._split_question(
        "Kit Rowan — Fraud over $5,000 (s.380(1)(a)) — party-liability basis (s.21)"
    )
    assert who == "Kit Rowan"
    assert what.startswith("Fraud over $5,000") and "party-liability" in what


def test_compare_falls_back_to_wording_when_no_statute_is_cited():
    ct = _compare_module()
    assert ct._same_question("X — Theft of a bicycle", "X — Theft of a bicycle") is True
    assert ct._same_question("X — Theft of a bicycle", "X — Arson of a warehouse") is False


def test_compare_trials_reads_a_single_accused_verdict(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("ct2", root / "scripts" / "compare_trials.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)

    t = tmp_path / "trial_y.md"
    t.write_text(
        "## The Verdict\n\n**Jury Foreperson:** Verdict: Not Guilty\n\n- Tally: Guilty: 0, Not Guilty: 3\n",
        encoding="utf-8",
    )
    assert ct._parse(t)["verdicts"] == {"verdict": "Not Guilty"}


def test_pinned_charges_hold_the_elements_fixed():
    # The elements of an offence are settled law. Pinning them is what stops the
    # intake clerk re-deriving a different list — and a different verdict — per run.
    c = CaseInput(
        title="t", charge_or_claim="fraud", your_side="s",
        charges=[Charge(label="Fraud over $5,000 (s.380(1)(a))", elements=["a", "b"])],
    )
    assert c.charges[0].label.startswith("Fraud over")
    assert c.charges[0].elements == ["a", "b"]
    assert CaseInput(title="t", charge_or_claim="c", your_side="s").charges == []


def test_reliability_defaults():
    c = CaseInput(title="t", charge_or_claim="c", your_side="s")
    assert c.verdict_passes == 1          # opt in to the extra cost
    assert c.strict_elements is True
    assert c.calibrated_proof is True
    assert c.evidence_digest is True
    assert c.deadlock_exhortation is True
    assert c.proof_threshold is None


def test_verdict_passes_is_bounded():
    import pytest
    with pytest.raises(Exception):
        CaseInput(title="t", charge_or_claim="c", your_side="s", verdict_passes=9)


def test_element_finding_probability_is_optional():
    assert ElementFinding(element="e").probability is None
    assert ElementFinding(element="e", probability=0).probability == 0


def test_probability_accepts_the_shapes_models_actually_emit():
    # A rejected value would fail the whole ballot and cost that juror their vote,
    # so every plausible spelling has to land somewhere sane.
    def p(v):
        return ElementFinding.model_validate({"element": "e", "probability": v}).probability

    assert p(85) == 85
    assert p("85") == 85
    assert p("85%") == 85
    assert p(" 90 % ") == 90
    assert p(0.85) == 85        # a 0-1 fraction where a percentage was asked for
    assert p(1) == 100          # ambiguous, but "certain" either way
    assert p(140) == 100        # clamped, not rejected
    assert p(-5) == 0
    assert p("high") is None    # unreadable -> fall back to the boolean finding
    assert p("") is None
    assert p(None) is None
    assert p(True) is None      # a stray boolean is not a probability


def test_unreadable_probability_leaves_the_finding_usable():
    f = ElementFinding.model_validate({"element": "e", "proven": True, "probability": "very high"})
    assert f.probability is None and _element_proven(f, 90) is True


# ---------------------------------------------------------------------------
# run_structured: transport failures, candidate ordering, concurrency ceiling
# ---------------------------------------------------------------------------
class _Reply:
    def __init__(self, output: str) -> None:
        self.output = output


class _FakeAgent:
    """Scripted agent: each entry is either a reply string or an exception to raise."""

    def __init__(self, script: list, *, hold: float = 0.0, tracker: dict | None = None) -> None:
        self.script = list(script)
        self.calls = 0
        self.hold = hold
        self.tracker = tracker

    async def run(self, prompt: str):
        import asyncio
        self.calls += 1
        if self.tracker is not None:
            self.tracker["inflight"] += 1
            self.tracker["peak"] = max(self.tracker["peak"], self.tracker["inflight"])
        try:
            if self.hold:
                await asyncio.sleep(self.hold)
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return _Reply(item)
        finally:
            if self.tracker is not None:
                self.tracker["inflight"] -= 1


def test_json_candidates_prefers_the_object_over_a_leading_bracket():
    from app.llm_utils import json_candidates
    # The old rule took whichever bracket came first and handed back '[1]: {"a": 1}'.
    cands = json_candidates('Note [1]: {"a": 1}')
    assert cands[0] == '{"a": 1}'
    assert json_candidates('```json\n{"b": 2}\n```') == ['{"b": 2}']
    assert json_candidates("no json here") == ["no json here"]


def test_run_structured_retries_a_transport_failure(monkeypatch):
    import asyncio
    import app.llm_utils as lu
    from app.schemas import Speech

    monkeypatch.setattr(lu, "_BACKOFF_SECONDS", 0.0)
    agent = _FakeAgent([ConnectionError("reset by peer"), '{"statement": "ok"}'])
    out = asyncio.run(lu.run_structured(agent, "p", Speech))
    assert out.statement == "ok" and agent.calls == 2


def test_run_structured_gives_up_after_the_retry_budget(monkeypatch):
    import asyncio
    import pytest
    import app.llm_utils as lu
    from app.schemas import Speech

    monkeypatch.setattr(lu, "_BACKOFF_SECONDS", 0.0)
    agent = _FakeAgent([TimeoutError("t"), TimeoutError("t"), TimeoutError("t")])
    with pytest.raises(ValueError):
        asyncio.run(lu.run_structured(agent, "p", Speech, retries=2))
    assert agent.calls == 3


def test_run_structured_falls_through_to_the_next_json_candidate():
    import asyncio
    import app.llm_utils as lu
    from app.schemas import Speech

    # A reply whose first {...} span is not the object we want, but whose second
    # candidate parses — no retry (and no extra model call) should be needed.
    agent = _FakeAgent(['Here you go [see 1]: {"statement": "yes"}'])
    out = asyncio.run(lu.run_structured(agent, "p", Speech))
    assert out.statement == "yes" and agent.calls == 1


def test_run_structured_bounds_concurrency(monkeypatch):
    import asyncio
    import app.llm_utils as lu
    from app.schemas import Speech

    monkeypatch.setattr(lu, "_MAX_CONCURRENCY", 3)
    tracker = {"inflight": 0, "peak": 0}

    async def go():
        agents = [_FakeAgent(['{"statement": "s"}'], hold=0.01, tracker=tracker) for _ in range(12)]
        return await asyncio.gather(*[lu.run_structured(a, "p", Speech) for a in agents])

    results = asyncio.run(go())
    assert len(results) == 12 and all(r.statement == "s" for r in results)
    assert tracker["peak"] <= 3  # never more than the ceiling in flight at once


# ---------------------------------------------------------------------------
# Headline vote in a multi-charge case follows the per-charge breakdown
# ---------------------------------------------------------------------------
def test_headline_from_charges_is_convict_on_any_count():
    from app.orchestrator import _headline_from_charges
    mixed = [ChargeVote(charge_label="Fraud", vote="convict"), ChargeVote(charge_label="Poss", vote="acquit")]
    clear = [ChargeVote(charge_label="Fraud", vote="acquit"), ChargeVote(charge_label="Poss", vote="acquit")]
    assert _headline_from_charges(mixed) == "convict"
    assert _headline_from_charges(clear) == "acquit"
    assert _headline_from_charges([]) == "acquit"

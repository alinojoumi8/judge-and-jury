"""Run a full trial from a JSON case file and save a readable transcript.

Usage (from the project root, with .env configured):
    python scripts/run_case_trial.py [path/to/case.json] [--jury N] [--rounds N]

Defaults to samples/example_case.json. --jury / --rounds override the values
in the JSON for a one-off run. Streams the trial to the console live and writes a
Markdown transcript to outputs/trial_<timestamp>.md as it goes, so the verdict and
ruling are captured even for a long run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Print UTF-8 to the Windows console (statements contain em-dashes / smart quotes).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.orchestrator import run_trial  # noqa: E402
from app.schemas import CaseInput  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an AI-courtroom trial from a JSON case file.")
    p.add_argument(
        "case",
        nargs="?",
        default=str(ROOT / "samples" / "example_case.json"),
        help="Path to the case JSON (default: samples/example_case.json).",
    )
    p.add_argument("--jury", type=int, default=None, help="Override jury size (1-12).")
    p.add_argument("--rounds", type=int, default=None, help="Override argument rounds (1-5).")
    p.add_argument(
        "--style", choices=["dialogue", "poll"], default=None,
        help="Override deliberation style: 'dialogue' (jurors debate) or 'poll' (quiet re-vote).",
    )
    p.add_argument(
        "--passes", type=int, default=None,
        help="Independent ballots per juror on the binding vote (1-5). Higher = less "
             "sampling noise in the verdict, at proportionally more cost.",
    )
    p.add_argument(
        "--no-digest", action="store_true",
        help="Skip the neutral per-element evidence digest before deliberation.",
    )
    p.add_argument(
        "--repeat", type=int, default=1,
        help="Run the same case N times and print a verdict-stability report — how "
             "often repeated runs actually agreed.",
    )
    p.add_argument("--label", default=None, help="Tag for the transcript filename (default: case stem).")
    return p.parse_args()


def _render_structured(speaker: str, data: dict) -> list[str]:
    """Turn a structured event's data into readable Markdown lines."""
    out: list[str] = []
    if "_manifest" in data:  # the settings this run was produced under
        out.append(
            f"- Model: `{data.get('model','')}` · jury {data.get('jury_size')} · "
            f"{data.get('argument_rounds')} argument round(s) · "
            f"{data.get('deliberation_rounds')} deliberation round(s) "
            f"({data.get('deliberation_style')})"
        )
        out.append(
            f"- Reliability: verdict_passes={data.get('verdict_passes')} · "
            f"proof threshold {data.get('proof_threshold')}% · "
            f"strict_elements={data.get('strict_elements')} · "
            f"calibrated_proof={data.get('calibrated_proof')} · "
            f"evidence_digest={data.get('evidence_digest')} · "
            f"straw_poll={data.get('straw_poll')} · "
            f"grounding_check={data.get('grounding_check')}"
        )
    elif "_digest" in data:  # the neutral per-element evidence digest
        for ce in data.get("charges", []):
            out.append(f"- **{ce.get('charge_label','')}**")
            for ee in ce.get("elements", []):
                out.append(f"  - _{ee.get('element','')}_")
                out.extend(f"    - ✔ {s}" for s in ee.get("supporting", []))
                out.extend(f"    - ✘ {u}" for u in ee.get("undermining", []))
                out.extend(f"    - ○ not in the record: {g}" for g in ee.get("gaps", []))
        if data.get("undisputed"):
            out.append("- Undisputed: " + "; ".join(data["undisputed"]))
        if data.get("disputed"):
            out.append("- In dispute: " + "; ".join(data["disputed"]))
    elif "_diagnostics" in data:  # ballot integrity for the deciding vote
        out.append(
            f"- Ballots counted: {data.get('ballots_counted')}/{data.get('ballots_expected')} "
            f"(deciding round {data.get('deciding_round')})"
        )
        agree = data.get("mean_sample_agreement")
        out.append(
            f"- Mean juror self-agreement: {int(agree * 100)}% over "
            f"{data.get('verdict_passes')} samples"
            if agree is not None
            else "- Ballots were single-sampled (not resampled) — run with --passes 3 to measure stability"
        )
        if data.get("abstentions"):
            out.append("- Abstained (excluded from the count): " + "; ".join(data["abstentions"]))
        if data.get("unmatched_entries"):
            out.append("- Ballot entries that had to fall back to a top-level vote:")
            out.extend(f"  - {g}" for g in data["unmatched_entries"])
    elif "_record" in data:  # the Agreed Record (source of truth)
        rec = data.get("agreed_record", {})
        if rec.get("parties"):
            out.append(f"- Parties: {'; '.join(rec['parties'])}")
        if rec.get("figures"):
            out.append(f"- Key figures: {'; '.join(rec['figures'])}")
        if rec.get("dates"):
            out.append(f"- Key dates: {'; '.join(rec['dates'])}")
        if rec.get("admissible_facts"):
            out.append("- Admissible facts:")
            out.extend(f"  - {f}" for f in rec["admissible_facts"])
        out.append(
            "- Authorities on record: "
            + ("; ".join(rec["authorities"]) if rec.get("authorities") else "none")
        )
    elif "_cast" in data:  # the auto-cast personalities for the non-jury roles
        for role in ("crown", "defense", "judge"):
            p = data.get(role) or {}
            if p.get("name") or p.get("style"):
                out.append(
                    f"- **{role.title()} — {p.get('name', '')}** · {p.get('background', '')} "
                    f"({p.get('style', '')})"
                )
        ws = data.get("witnesses") or []
        if ws:
            out.append("- Witnesses:")
            out.extend(
                f"  - **{w.get('name', '')}** — {w.get('style', '')}" for w in ws
            )
    elif "_straw" in data:  # pre-discussion straw poll (a Verdict payload)
        tally = ", ".join(f"{k}: {v}" for k, v in data.get("tally", {}).items())
        out.append(f"- Straw poll (private, before discussion): {tally}")
    elif "_movement" in data:  # how the room moved after deliberation
        i = ", ".join(f"{k}: {v}" for k, v in data.get("initial_tally", {}).items())
        f = ", ".join(f"{k}: {v}" for k, v in data.get("final_tally", {}).items())
        out.append(f"- Straw poll: {i}  →  Final: {f}")
        if data.get("flips"):
            out.append("- Jurors who moved:")
            out.extend(f"  - {x}" for x in data["flips"])
        else:
            out.append("- No jurors changed their vote during deliberation.")
    elif "_grounding" in data:  # fact-check flags
        out.append("> ⚠ Fact-check flags:")
        for f in data.get("flags", []):
            out.append(
                f"  - [{f.get('severity', '')}/{f.get('issue', '')}] "
                f"{f.get('claim', '')} — {f.get('explanation', '')}"
            )
    elif "_directed_verdict" in data:  # directed-verdict ruling
        out.append("- " + ("Directed verdict GRANTED" if data.get("granted") else "Motion dismissed"))
        if data.get("reasoning"):
            out.append(f"  - {data['reasoning']}")
        if data.get("per_defendant"):
            out.append(f"  - Acquitted: {'; '.join(data['per_defendant'])}")
    elif "jurors" in data:  # jury selection
        for j in data["jurors"]:
            out.append(f"- **{j.get('name','Juror')}** — {j.get('background','')} "
                       f"({j.get('disposition','')})")
    elif "outcome" in data:  # final verdict (overall, per-defendant, or per-charge)
        if data.get("defendant_name") and data.get("charge_label"):
            out.append(f"- **{data['defendant_name']} — {data['charge_label']}: {data['outcome']}**")
        elif data.get("charge_label"):
            out.append(f"- **{data['charge_label']}: {data['outcome']}**")
        tally = ", ".join(f"{k}: {v}" for k, v in data.get("tally", {}).items())
        out.append(f"- Tally: {tally}")
        out.append(f"- Unanimous: {data.get('unanimous')} · Hung: {data.get('hung')}")
        if data.get("abstentions"):
            out.append(f"- Abstentions (excluded from the tally): {data['abstentions']}")
        out.append(f"- {data.get('dissent_summary','')}")
    elif "sentence_or_remedy" in data:  # judge's ruling
        if data.get("verdict_acknowledgement"):
            out.append(f"- _Acknowledgement:_ {data['verdict_acknowledgement']}")
        if data.get("reasoning"):
            out.append(f"- _Reasoning:_ {data['reasoning']}")
        if data.get("sentence_or_remedy"):
            out.append(f"- _Sentence/remedy:_ {data['sentence_or_remedy']}")
        if data.get("aggravating_factors"):
            out.append("- _Aggravating:_ " + "; ".join(data["aggravating_factors"]))
        if data.get("mitigating_factors"):
            out.append("- _Mitigating:_ " + "; ".join(data["mitigating_factors"]))
        if data.get("sentencing_range"):
            out.append(f"- _Range:_ {data['sentencing_range']}")
        if data.get("restitution"):
            out.append(f"- _Restitution:_ {data['restitution']}")
        if data.get("conditions"):
            out.append("- _Conditions:_ " + "; ".join(data["conditions"]))
        if data.get("closing_remarks"):
            out.append(f"- _Closing:_ {data['closing_remarks']}")
    elif "verdict" in data and "reasoning" in data:  # a juror's vote
        def _elements(findings: list, indent: str) -> None:
            for ef in findings or []:
                mark = "PROVEN" if ef.get("proven") else "not proven"
                p = ef.get("probability")
                pct = f" ({p}%)" if isinstance(p, int) else ""
                out.append(f"{indent}- [{mark}{pct}] {ef.get('element', '')}")

        def _charges(cvs: list, indent: str) -> None:
            for cv in cvs or []:
                out.append(f"{indent}- **{cv.get('charge_label', '?')}**: {cv.get('verdict', '')}")
                if cv.get("reasoning"):
                    out.append(f"{indent}  - _{cv['reasoning']}_")
                _elements(cv.get("element_findings"), indent + "  ")

        if data.get("defendant_votes"):  # multi-accused: one block per co-accused
            for dv in data["defendant_votes"]:
                out.append(f"  - **{dv.get('defendant_name', '?')}** — {dv.get('verdict', '')}")
                if dv.get("charge_votes"):
                    _charges(dv.get("charge_votes"), "    ")
                else:
                    if dv.get("reasoning"):
                        out.append(f"    - _{dv['reasoning']}_")
                    _elements(dv.get("element_findings"), "    ")
        elif data.get("charge_votes"):  # single accused, multiple charges
            _charges(data.get("charge_votes"), "  ")
        else:
            if data.get("reasoning"):
                out.append(f"  - _{data['reasoning']}_")
            _elements(data.get("element_findings"), "  ")
    elif "theory" in data and "strongest_points" in data:  # counsel's case strategy
        if data.get("theory"):
            out.append(f"- Theory: {data['theory']}")
        if data.get("strongest_points"):
            out.append("- Strongest points:")
            out.extend(f"  - {p}" for p in data["strongest_points"])
        if data.get("opponents_best_point"):
            out.append(f"- Opponent's best point: {data['opponents_best_point']}")
        if data.get("rebuttal"):
            out.append(f"- Planned rebuttal: {data['rebuttal']}")
    elif "prosecution_theory" in data:  # intake structured case
        if data.get("charges_or_claims"):
            out.append(f"- Charges/claims: {'; '.join(data['charges_or_claims'])}")
        if data.get("prosecution_theory"):
            out.append(f"- Prosecution theory: {data['prosecution_theory']}")
        if data.get("defense_theory"):
            out.append(f"- Defense theory: {data['defense_theory']}")
        if data.get("key_facts"):
            out.append("- Key facts:")
            out.extend(f"  - {f}" for f in data["key_facts"])
        if data.get("elements"):
            out.append("- Elements the prosecution must prove:")
            out.extend(f"  - {e}" for e in data["elements"])
    return out


def _verdict_key(ev) -> str | None:
    """The name of the thing this verdict event decides, or None if it isn't one."""
    d = ev.data or {}
    if "outcome" not in d or "_straw" in d or "_movement" in d:
        return None
    if d.get("defendant_name") and d.get("charge_label"):
        return f"{d['defendant_name']} — {d['charge_label']}"
    if d.get("charge_label"):
        return d["charge_label"]
    if d.get("defendant_name"):
        return d["defendant_name"]
    return "verdict"


async def _run_one(case: CaseInput, out_file: Path) -> dict:
    """Stream one trial to console + transcript; return its outcomes for comparison."""
    lines: list[str] = []
    outcomes: dict[str, str] = {}

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)
        out_file.write_text("\n".join(lines), encoding="utf-8")

    emit(f"# Trial transcript — {case.title}")
    emit(f"_{case.case_type} · {case.jurisdiction} · jury of {case.jury_size} · "
         f"{case.argument_rounds} argument round(s)_")
    emit("")
    emit(f"**Charge / claim:** {case.charge_or_claim}")

    async for ev in run_trial(case):
        if ev.phase == "Verdict" and ev.kind == "structured":
            key = _verdict_key(ev)
            if key:
                outcomes[key] = (ev.data or {}).get("outcome", "")
        if ev.kind == "phase":
            emit("")
            emit(f"## {ev.content}")
        elif ev.kind in ("message", "structured"):
            emit("")
            emit(f"**{ev.speaker}:** {ev.content}")
            if ev.kind == "structured" and ev.data:
                emit("")
                for ln in _render_structured(ev.speaker, ev.data):
                    emit(ln)
        elif ev.kind == "error":
            emit("")
            emit(f"> **ERROR:** {ev.content}")
            outcomes.setdefault("_error", ev.content)
        elif ev.kind == "done":
            emit("")
            emit("_Trial complete._")

    print(f"\n[saved transcript -> {out_file}]", flush=True)
    return outcomes


def _stability_report(runs: list[dict]) -> list[str]:
    """How often repeated runs of the same case agreed — the reliability measure.

    A simulator whose verdict swings run to run is telling you about the sampler,
    not about the case, so this is worth seeing plainly rather than inferring from
    two transcripts side by side.
    """
    keys: list[str] = []
    for r in runs:
        for k in r:
            if k != "_error" and k not in keys:
                keys.append(k)
    out = ["", f"## Verdict stability across {len(runs)} run(s)", ""]
    out.append("| Question | " + " | ".join(f"Run {i + 1}" for i in range(len(runs))) + " | Agreement |")
    out.append("|---|" + "---|" * (len(runs) + 1))
    stable = 0
    for k in keys:
        vals = [r.get(k, "—") for r in runs]
        agreed = len(set(vals)) == 1
        stable += agreed
        out.append(f"| {k} | " + " | ".join(vals) + " | " + ("identical" if agreed else "**DIVERGED**") + " |")
    if keys:
        out.append("")
        out.append(f"**{stable}/{len(keys)} questions returned the same verdict in every run.**")
    return out


async def main() -> None:
    args = _parse_args()
    data = json.loads(Path(args.case).read_text(encoding="utf-8"))
    if args.jury is not None:
        data["jury_size"] = args.jury
    if args.rounds is not None:
        data["argument_rounds"] = args.rounds
    if args.style is not None:
        data["deliberation_style"] = args.style
    if args.passes is not None:
        data["verdict_passes"] = args.passes
    if args.no_digest:
        data["evidence_digest"] = False
    case = CaseInput(**data)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or Path(args.case).stem

    runs: list[dict] = []
    for i in range(max(1, args.repeat)):
        suffix = f"_r{i + 1}" if args.repeat > 1 else ""
        # PID keeps concurrent runs (same second) from overwriting each other.
        out_file = out_dir / f"trial_{stamp}_{label}{suffix}_{os.getpid()}.md"
        if args.repeat > 1:
            print(f"\n===== RUN {i + 1} of {args.repeat} =====\n", flush=True)
        runs.append(await _run_one(case, out_file))

    if len(runs) > 1:
        report = _stability_report(runs)
        print("\n".join(report), flush=True)
        report_file = out_dir / f"stability_{stamp}_{label}_{os.getpid()}.md"
        report_file.write_text(
            f"# Verdict stability — {case.title}\n"
            f"_{args.repeat} identical runs · verdict_passes={case.verdict_passes} · "
            f"jury of {case.jury_size}_\n" + "\n".join(report),
            encoding="utf-8",
        )
        print(f"\n[saved stability report -> {report_file}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

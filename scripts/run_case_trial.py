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
    p.add_argument("--label", default=None, help="Tag for the transcript filename (default: case stem).")
    return p.parse_args()


def _render_structured(speaker: str, data: dict) -> list[str]:
    """Turn a structured event's data into readable Markdown lines."""
    out: list[str] = []
    if "jurors" in data:  # jury selection
        for j in data["jurors"]:
            out.append(f"- **{j.get('name','Juror')}** — {j.get('background','')} "
                       f"({j.get('disposition','')})")
    elif "outcome" in data:  # final verdict
        tally = ", ".join(f"{k}: {v}" for k, v in data.get("tally", {}).items())
        out.append(f"- Tally: {tally}")
        out.append(f"- Unanimous: {data.get('unanimous')} · Hung: {data.get('hung')}")
        out.append(f"- {data.get('dissent_summary','')}")
    elif "sentence_or_remedy" in data:  # judge's ruling
        if data.get("verdict_acknowledgement"):
            out.append(f"- _Acknowledgement:_ {data['verdict_acknowledgement']}")
        if data.get("reasoning"):
            out.append(f"- _Reasoning:_ {data['reasoning']}")
        if data.get("sentence_or_remedy"):
            out.append(f"- _Sentence/remedy:_ {data['sentence_or_remedy']}")
        if data.get("closing_remarks"):
            out.append(f"- _Closing:_ {data['closing_remarks']}")
    elif "verdict" in data and "reasoning" in data:  # a single juror's vote
        if data.get("reasoning"):
            out.append(f"  - _{data['reasoning']}_")
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
    return out


async def main() -> None:
    args = _parse_args()
    data = json.loads(Path(args.case).read_text(encoding="utf-8"))
    if args.jury is not None:
        data["jury_size"] = args.jury
    if args.rounds is not None:
        data["argument_rounds"] = args.rounds
    case = CaseInput(**data)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or Path(args.case).stem
    # PID keeps concurrent runs (same second) from overwriting each other.
    out_file = out_dir / f"trial_{stamp}_{label}_{os.getpid()}.md"

    lines: list[str] = []

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
        elif ev.kind == "done":
            emit("")
            emit("_Trial complete._")

    print(f"\n[saved transcript -> {out_file}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

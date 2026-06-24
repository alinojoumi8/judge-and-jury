"""End-to-end smoke test: run a tiny trial in the terminal (no web UI).

Usage (from the project root, with .env configured):
    python scripts/smoke_test.py

Confirms the MiniMax key/model work and the full orchestration completes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Print UTF-8 to the Windows console (statements contain em-dashes / smart quotes).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Allow running as a plain script: add the project root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator import run_trial  # noqa: E402
from app.schemas import CaseInput  # noqa: E402

SAMPLE = CaseInput(
    title="The Case of the $5 Chocolate Bar",
    case_type="criminal",
    jurisdiction="Ontario, Canada",
    charge_or_claim="Theft under $5,000 (shoplifting a chocolate bar)",
    your_side=(
        "My client says she picked up a chocolate bar, got distracted by a phone "
        "call, and walked out still holding it without realising she hadn't paid. "
        "She returned to pay as soon as she noticed. There was no intent to steal."
    ),
    jury_size=3,
    argument_rounds=1,
)


async def main() -> None:
    print(f"Running trial: {SAMPLE.title}\n" + "=" * 60)
    async for ev in run_trial(SAMPLE):
        if ev.kind == "phase":
            print(f"\n=== {ev.content} ===")
        elif ev.kind == "speaker_start":
            print(f"\n[{ev.speaker}] ", end="", flush=True)
        elif ev.kind == "delta":
            print(ev.content, end="", flush=True)
        elif ev.kind == "message":
            print(f"\n[{ev.speaker}] {ev.content}")
        elif ev.kind == "structured":
            print(f"\n[{ev.speaker}] {ev.content}")
        elif ev.kind == "error":
            print(f"\n!! ERROR: {ev.content}")
        elif ev.kind == "done":
            print("\n\n" + "=" * 60 + "\nTrial complete.")


if __name__ == "__main__":
    asyncio.run(main())

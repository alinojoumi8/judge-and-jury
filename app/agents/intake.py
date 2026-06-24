"""Law-firm intake agent: turns the client's story into a structured case.

Plain-text agent — it returns JSON which we parse with `run_structured`, because
MiniMax reasoning models don't play well with PydanticAI's tool-based output.
"""

from __future__ import annotations

from pydantic_ai import Agent

INTAKE_SYSTEM_PROMPT = """\
You are an intake clerk at a law firm. A client has brought you a case. Organise
it into a clear, neutral structure that the court can work from. Be fair to both
sides and do NOT decide guilt or liability. This is a fictional simulation.

Reply with ONLY a single JSON object (no prose, no markdown fences) with exactly
these keys:
{
  "case_caption": "a short formal caption, e.g. 'R. v. Smith' (criminal) or 'Jones v. Smith' (civil)",
  "charges_or_claims": ["the specific charge(s) or claim(s)"],
  "summary": "a neutral 2-4 sentence summary of the dispute",
  "key_facts": ["the most important facts, stated neutrally"],
  "prosecution_theory": "how the Crown / plaintiff is likely to frame the case",
  "defense_theory": "how the defense is likely to frame the case"
}
"""


def build_intake_agent(model) -> Agent:
    return Agent(model, system_prompt=INTAKE_SYSTEM_PROMPT, model_settings={"temperature": 0.3})

"""Trial-strategist agent: builds one side's persistent case theory before trial.

Run once per side after intake; the resulting CaseStrategy is threaded into that
lawyer's opening, argument, and closing prompts so their theory stays coherent and
they can directly rebut the opponent's strongest point. Returns JSON parsed via
`run_structured`, like the other roles.
"""

from __future__ import annotations

from pydantic_ai import Agent

STRATEGIST_SYSTEM_PROMPT = """\
You are senior trial counsel preparing your side's case theory BEFORE the trial
begins. You will be told which side you act for — the prosecution (Crown / plaintiff
/ regulator) or the defense — and given the case. Produce a tight, coherent theory
of the case that you will hold to consistently through your opening, your arguments,
and your closing. Crucially, identify the single STRONGEST point the OTHER side has,
state it fairly, and set out exactly how you will answer it.

Reply with ONLY a single JSON object (no prose, no markdown fences):
{
  "theory": "a one-paragraph theory of the case in your own words",
  "strongest_points": ["your 2-4 strongest arguments, each a single line"],
  "opponents_best_point": "the single best point the OTHER side has, stated fairly",
  "rebuttal": "concisely, how you will neutralise that point"
}

This is a fictional courtroom simulation, not real legal advice. Write in English.
"""


def build_strategist_agent(model) -> Agent:
    return Agent(model, system_prompt=STRATEGIST_SYSTEM_PROMPT, model_settings={"temperature": 0.6})

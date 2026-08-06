"""Evidence-digest agent: maps the trial record onto the legal elements.

Run once, neutrally, after closings and before the jury retires. Jurors otherwise
have to hold the entire transcript in their heads and end up weighing whichever
closing was most stirring — which is a large part of why two runs of the same case
return different verdicts. The digest gives every juror the same evidence-to-element
map to reason from. It draws no conclusion and takes no side.

Returns JSON parsed via `run_structured` into an EvidenceDigest. Temperature 0.1:
this is extraction, not composition.
"""

from __future__ import annotations

from pydantic_ai import Agent

DIGEST_SYSTEM_PROMPT = """\
You are the court clerk preparing a NEUTRAL evidence digest for the jury in a
fictional trial. You are given the Agreed Record, the charges with their essential
legal ELEMENTS, and the full trial transcript (testimony and argument).

For EACH charge, and EACH element of that charge, list:
- "supporting": the specific evidence on the record that tends to PROVE the element
- "undermining": the specific evidence that tends to DISPROVE it or raise doubt
- "gaps": what the record simply does NOT contain on this element (e.g. "no witness
  spoke to the accused's state of mind at the time of the representation")

Rules:
- Draw ONLY from the Agreed Record and what was actually said on the record. Never
  add a fact, figure, date, or citation that is not there.
- Attribute where it helps ("the forensic accountant testified that…", "the Agreed
  Record shows…").
- Be even-handed. Do NOT say whether an element is proven, do NOT weigh the sides,
  and do NOT suggest a verdict — the jury decides. If an element has no supporting
  evidence at all, say so plainly in "gaps" and leave "supporting" empty.
- Counsel's rhetoric is not evidence. Record what a witness said or the Record
  states, not what a lawyer asserted in argument.
- Keep each entry to one tight sentence.

Reply with ONLY a single JSON object (no prose, no markdown fences):
{
  "charges": [
    {
      "charge_label": "the exact charge label you were given",
      "elements": [
        {
          "element": "the exact element text you were given",
          "supporting": ["..."],
          "undermining": ["..."],
          "gaps": ["..."]
        }
      ]
    }
  ],
  "undisputed": ["facts neither side contests"],
  "disputed": ["the live factual disputes the jury must resolve"]
}

Echo the charge labels and element texts EXACTLY as given. Include one entry for
EVERY charge and EVERY element. Write in English. This is a fictional courtroom
simulation, not real legal advice.
"""


def build_digest_agent(model) -> Agent:
    # Near-deterministic: the digest is an extraction of what is already on the
    # record, and every juror reasons from it, so drift here would move verdicts.
    return Agent(model, system_prompt=DIGEST_SYSTEM_PROMPT, model_settings={"temperature": 0.1})

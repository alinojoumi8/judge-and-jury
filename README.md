# ⚖️ AI Courtroom — Judge & Jury

A multi-agent **trial simulator**. You bring a case (as the client of a law firm),
and AI agents play it out: a **Defense lawyer** argues for you, the **Crown**
(prosecutor) argues against, a configurable **jury** listens and votes, and a
**judge** presides and delivers the ruling — all streaming live in your browser.

Built with **PydanticAI** agents powered by **MiniMax** models, served by
**FastAPI** with live (SSE) streaming.

> ⚠️ **This is an educational/entertainment simulation, not legal advice.** It does
> not reflect any real court, law, person, or outcome.

---

## How it works

Each courtroom role is a PydanticAI agent with its own system prompt, all driven by
your MiniMax model. A state machine ([`app/orchestrator.py`](app/orchestrator.py))
runs the trial and streams events to the page:

1. **Intake** — a law-firm clerk turns your story into a structured case and settles
   an **Agreed Record**: an immutable ledger of the parties, key figures, dates,
   admissible facts, and the only legal authorities in play. It is threaded into
   every later prompt as the single source of truth, so agents argue from it rather
   than inventing facts or case citations.
2. **Jury selection & casting** — generates your chosen number of distinct jurors,
   and (by default) **auto-casts a personality for the Crown, Defence, Judge, and
   each witness** — a name, background, and manner (advocacy style / bench
   temperament / witness demeanour) that each role then holds consistently, so the
   trial reads like distinct real people rather than interchangeable archetypes.
   Turn it off with `personas: false`.
3. **Judge opening** — frames the case, sets out the **legal elements** the Crown
   must prove, and charges the jury that a reasonable doubt on *any one* element
   means acquittal.
4. **Opening statements** — before they speak, each side privately fixes a **case
   theory** (its strongest points plus the opponent's best argument and how it will
   answer it) that it then holds to consistently through the trial. Crown opens,
   then Defense.
5. **Witness testimony** *(optional)* — each witness is examined, cross-examined, and
   **re-examined**. Witnesses **remember their own prior testimony** (carried across
   direct → cross → re-direct), so cross-examination can probe inconsistencies; either
   side may object on any examination and the judge rules. Once the prosecution closes
   its case the defence may move for a **directed verdict** (no-evidence motion), which
   the judge grants or dismisses. Skipped if you supply no witnesses.
6. **Argument rounds** — Crown vs. Defense, back and forth (configurable count).
   Each turn opens by **steel-manning the opponent's strongest point and rebutting
   it head-on** before advancing its own case; closings do the same.
7. **Closing statements** — Crown, then Defense.
8. **Jury deliberation** — the jury first takes a **private straw poll** (each juror
   votes independently, before any discussion, to curb herding). Then, by default, it
   holds a **spoken deliberation**: a foreperson opens, each juror speaks in turn,
   hearing and responding to the others, before casting a binding vote. Jurors decide
   **element by element** — convicting only if *every* element is proven — and a
   low-confidence consensus is sent back for another round. The transcript shows how
   the room moved from the straw poll. Set `deliberation_style: "poll"` for the
   classic quiet parallel re-vote.
9. **Verdict** — tallied from the votes. **Criminal verdicts require unanimity** (any
   split is a hung jury → mistrial); civil verdicts carry on a majority. With multiple
   **co-accused**, each defendant gets a separate verdict; with multiple **charges**,
   each charge gets its own verdict (e.g. guilty of fraud, not guilty of possession).
10. **Judge's ruling** — sentence (criminal) or remedy (civil); a mistrial if hung. On
    a conviction the sentence is **structured** — aggravating/mitigating factors, a
    realistic range, restitution, and conditions.

> **Anti-hallucination fact-check** *(opt-in)*: with `grounding_check: true`, a neutral
> verifier scans the most consequential statements (closings and the ruling by default)
> against the Agreed Record and flags any ungrounded claim, misquoted figure, or
> invented case citation — shown inline, never altering the verdict.

> **Optional inputs**: `defendants` (co-accused, each judged separately), `witnesses`
> (adds the evidence phase), `deliberation_rounds`, `deliberation_style`
> (`"dialogue"` (default) / `"poll"`), `straw_poll` (default on), `personas`
> (auto-cast counsel/bench/witness personalities, default on; each `Witness` may also
> set a manual `demeanour`; **pin** a role with `crown_persona` / `defense_persona` /
> `judge_persona` — a `{name, background, style}` object — to A/B how advocacy style
> alone moves a verdict, see [`samples/personas_ab.json`](samples/personas_ab.json)),
> `redirect` /
> `qa_redirect` (re-examination, default on), `allow_directed_verdict` (default on),
> and the anti-hallucination knobs `grounding_check` / `grounding_phases` /
> `grounding_adversarial` / `self_ground` (all default off). Multiple charges are
> driven by the charge text — name more than one offence and each gets its own
> verdict. See [`samples/example_case_full.json`](samples/example_case_full.json) for a
> worked example, and run it with `python scripts/run_case_trial.py samples/example_case_full.json`.

---

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# then open .env and paste your MiniMax token-plan key
```

Edit `.env`:

```
MINIMAX_API_KEY=your-key-here
MINIMAX_BASE_URL=https://api.minimax.io/v1      # China: https://api.minimaxi.chat/v1
MINIMAX_MODEL=MiniMax-M2.1                       # set to a model your plan exposes
```

> **Model name matters.** Use a model id your MiniMax token plan actually exposes
> (e.g. `MiniMax-M2.1`, `MiniMax-M2.1-lightning`, `MiniMax-M2`, `MiniMax-Text-01`).
> If you get an auth/model error, that's usually the cause.

---

## Run

**Quick check (terminal, no UI):**

```powershell
python scripts/smoke_test.py
```

This runs a tiny trial (3 jurors, 1 round) and prints it — the fastest way to
confirm your key and model work end to end.

**The web app:**

```powershell
uvicorn app.main:app --reload
```

Open <http://localhost:8000>, fill in the case form, and click **Start Trial**.
You can sanity-check config at <http://localhost:8000/api/health>.

---

## Notes

- **Token cost** scales with **jury size × argument rounds**. Defaults are modest
  (3 jurors, 2 rounds) — turn them up once you're happy.
- **Criminal vs. civil**: wording adapts automatically
  (guilty/not guilty vs. liable/not liable; sentence vs. remedy).
- **MiniMax reasoning models** (M2.x) emit a `<think>…</think>` block before their
  answer. Every role returns JSON, which we strip + validate in
  [`app/llm_utils.py`](app/llm_utils.py); the browser reveals each statement with a
  typewriter effect for a live courtroom feel.
- **Project layout**:
  - [`app/agents/`](app/agents) — the role agents (intake, crown, defense, juror, judge)
  - [`app/orchestrator.py`](app/orchestrator.py) — the trial flow
  - [`app/main.py`](app/main.py) — FastAPI + SSE
  - [`web/`](web) — the courtroom UI (no build step)

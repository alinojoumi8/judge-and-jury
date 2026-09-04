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
8. **Evidence digest** — before the jury retires, a neutral clerk maps the record onto
   the elements: for each element of each charge, what supports it, what undercuts it,
   and what the record simply lacks. Every juror deliberates from the same map. See
   [Making the verdict reliable](#making-the-verdict-reliable).
9. **Jury deliberation** — the jury first takes a **private straw poll** (each juror
   votes independently, before any discussion, to curb herding). Then, by default, it
   holds a **spoken deliberation**: a foreperson opens, each juror speaks in turn,
   hearing and responding to the others, before casting a binding vote. Jurors decide
   **element by element** — convicting only if *every* element is proven — and a
   low-confidence consensus is sent back for another round. The transcript shows how
   the room moved from the straw poll. Set `deliberation_style: "poll"` for the
   classic quiet parallel re-vote. If a divided jury stops moving, the judge gives one
   **exhortation** and they try again.
10. **Verdict** — tallied from the votes. **Criminal verdicts require unanimity** (any
    split is a hung jury → mistrial); civil verdicts carry on a majority. With multiple
    **co-accused**, each defendant gets a separate verdict; with multiple **charges**,
    each charge gets its own verdict (e.g. guilty of fraud, not guilty of possession).
    A **ballot integrity** line reports how many ballots were counted and how stable
    they were.
11. **Judge's ruling** — sentence (criminal) or remedy (civil); a mistrial if hung. On
    a conviction the sentence is **structured** — aggravating/mitigating factors, a
    realistic range, restitution, and conditions. Nothing is sentenced that was not
    proved on some count.

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
> the anti-hallucination knobs `grounding_check` / `grounding_phases` /
> `grounding_adversarial` / `self_ground` (all default off), and the reliability knobs
> `verdict_passes` / `strict_elements` / `calibrated_proof` / `proof_threshold` /
> `evidence_digest` / `deadlock_exhortation`
> (see [Making the verdict reliable](#making-the-verdict-reliable)). Multiple charges are
> driven by the charge text — name more than one offence and each gets its own
> verdict. See [`samples/example_case_full.json`](samples/example_case_full.json) for a
> worked example, and run it with `python scripts/run_case_trial.py samples/example_case_full.json`.

---

## Making the verdict reliable

A courtroom simulator is only worth anything if the same case gives you roughly the
same answer twice. Early runs of the same fraud trial returned straw polls of 6–6,
3–9 and 1–11 — the verdict was largely a coin flip. These are the mechanisms that
close that gap, all on by default except the last.

Measured on a 12-juror criminal trial with three co-accused and two charges (six
verdicts per run), same case file and same model throughout:

| | Straw poll | Verdicts identical across runs |
|---|---|---|
| Before (3 runs) | 6–6, 3–9, 1–11 | **2 / 6 — 33%** |
| After (4 runs) | 0–12 in all four | **6 / 6 — 100%** |

The mechanism is visible in the ballots: with `verdict_passes: 3`, one run found 10 of
12 jurors perfectly self-consistent and **2 who changed their own answer between
samples**. On a single sample those two were coin flips — and a criminal verdict needs
unanimity, so one flipped juror is enough to hang the jury. That is what the 6–6 run
was.

**Pin the elements for any case you'll run twice.** Left to itself the intake clerk
derives the essential legal elements from your charge text — and derives a *different*
list next run. Since conviction is an AND across that list, changing it changes the
verdict. The elements of a given offence are settled law, so state them once in the
case file and they are used verbatim:

```json
"charges": [
  {"label": "Fraud over $5,000 (s.380(1)(a))",
   "elements": ["a dishonest act (deceit, falsehood or other fraudulent means)",
                "deprivation, or a real risk of deprivation, caused by that act",
                "the accused's subjective knowledge that the act was dishonest",
                "the accused's subjective knowledge that deprivation could follow"]},
  {"label": "Possession of proceeds of crime (s.354(1))",
   "elements": ["the accused possessed the property",
                "the property was derived from the commission of an indictable offence",
                "the accused knew the property was so derived"]}
]
```

**The verdict comes from the elements, checked against the real element list.** A
juror returns a finding per essential element, and the accused is convicted only if
*every* element is proven. Those findings are then aligned to the charge's actual
elements by meaning, not by exact string — so a reworded element still counts, an
element the juror never addressed counts as **not proven** (`strict_elements`, the
burden sits with the prosecution), and an element the juror invented is ignored.

**The standard of proof is enforced, not assumed.** Each finding carries a
`probability` (0–100). With `calibrated_proof` on, an element marked proven but
scored below the threshold — 90% for "beyond a reasonable doubt", 51% for "balance of
probabilities", override with `proof_threshold` — is treated as not proven. It only
ever *downgrades* an over-confident finding; it can never turn "not proven" into a
finding against the accused. This is what stops each juror privately inventing their
own idea of reasonable doubt.

**A failed ballot abstains — it never votes.** If the model cannot produce a valid
ballot for a juror, that juror is excluded from the count and named in the
transcript. Counting the failure as an acquittal (the old behaviour) meant a single
parse error could hang a criminal jury on its own.

**Diversity comes from personas, not from the sampler.** The juror agent runs near
the bottom of the temperature range; the *jury pool* generator stays high. Jurors
should disagree because they are different people reading the same record, not
because the sampler rolled differently.

**Deliberation is anchored to evidence.** Before the jury retires, a neutral clerk
compiles an **evidence digest**: for every element of every charge, what on the
record supports it, what undercuts it, and what the record simply does not contain.
Every juror reasons from the same map instead of from whichever closing was more
stirring. Turn it off with `evidence_digest: false`.

**Order effects are rotated out.** Whoever speaks first frames the room, so the
speaking (and vote-reporting) order rotates every deliberation round.

**A deadlocked jury is sent back once.** When a divided jury stops moving, the judge
delivers a single exhortation — reconsider with an open mind, but never surrender an
honestly held view to reach a verdict — and one extra round runs. A second stall is a
genuine hung jury. Off with `deadlock_exhortation: false`.

**Nothing is sentenced that was not proved.** Aggravating/mitigating factors, range,
restitution and conditions are stripped unless something actually resulted in a
conviction on some count, for some accused.

**Every run says what produced it.** A run manifest (model, jury size, rounds,
thresholds, every reliability flag) is emitted before intake, and a **ballot
integrity** report before the verdict: ballots counted vs expected, who abstained,
mean sample agreement, and any ballot entry that had to fall back to a top-level
vote.

**Self-consistency** *(opt-in — this one costs money)*: `verdict_passes: 3` has each
juror cast three independent ballots on the binding vote and counts their **modal**
one, reporting how often they agreed with themselves. This is the single most
effective lever against a close case being decided by sampling noise, at
proportionally more tokens on the final round only.

```powershell
python scripts/run_case_trial.py samples/example_case_full.json --passes 3
```

**Measure it, don't assume it.** `--repeat N` runs the same case N times and prints a
verdict-stability table — every question the jury answered, what each run returned,
and which ones diverged. That number is the honest measure of how much to trust a
single transcript.

```powershell
python scripts/run_case_trial.py samples/example_case_full.json --passes 3 --repeat 3
```

For transcripts produced separately — different sessions, or a before/after across a
code change — compare them after the fact:

```powershell
python scripts/compare_trials.py outputs/trial_A.md outputs/trial_B.md
```

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
# LLM_MAX_CONCURRENCY=8                          # optional: cap on simultaneous model calls
# LLM_RETRY_BACKOFF=1.5                          # optional: seconds before retrying a failed call
```

Model calls are bounded to `LLM_MAX_CONCURRENCY` at a time and retried with backoff
on transport failures, so a 12-juror, 5-pass deliberation (60 ballots at once)
stays under a token plan's rate limit instead of losing jurors to 429s.

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

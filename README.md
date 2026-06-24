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

1. **Intake** — a law-firm clerk turns your story into a structured case.
2. **Jury selection** — generates your chosen number of distinct jurors.
3. **Judge opening** — frames the case and instructs the jury.
4. **Opening statements** — Crown, then Defense.
5. **Argument rounds** — Crown vs. Defense, back and forth (configurable count).
6. **Closing statements** — Crown, then Defense.
7. **Jury deliberation** — each juror votes; a split jury reconsiders once.
8. **Verdict** — tallied from the jurors' votes.
9. **Judge's ruling** — sentence (criminal) or remedy (civil).

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

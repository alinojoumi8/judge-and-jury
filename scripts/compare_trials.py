"""Compare the verdicts of two or more saved trial transcripts.

`run_case_trial.py --repeat N` reports stability for runs it launched itself. This
does the same for transcripts produced separately — different sessions, different
machines, or a before/after comparison across a code change.

Usage (from the project root):
    python scripts/compare_trials.py outputs/trial_A.md outputs/trial_B.md
    python scripts/compare_trials.py outputs/trial_2026*_osc_*.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator import _similarity  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Two runs can name the same question differently — the intake clerk writes
# "Fraud over $5,000 (s.380(1)(a)) — party-liability basis (s.21)" one run and
# "…, via party liability under s.21" the next. Keying on the exact string would
# report two identical verdicts as a total disagreement, which is worse than
# useless. (Pinning `charges` on the case file avoids the drift at source — see the
# README.) Whole-string similarity is too blunt here: every row shares the same
# "party-liability (s.21)" boilerplate, which drowns out the part that actually
# distinguishes them. So compare the accused and the charge separately.
_SAME_ACCUSED = 0.6
_SAME_CHARGE = 0.6
_STATUTE = re.compile(r"s\.\s*(\d+)")


def _split_question(label: str) -> tuple[str, str]:
    """Split "Accused — Charge (s.NNN)" into its two halves.

    Splits on the FIRST separator only: a charge label may itself contain one
    ("Fraud over $5,000 (s.380(1)(a)) — party-liability basis (s.21)").
    """
    for sep in (" — ", " - ", ": "):
        if sep in label:
            head, _, tail = label.partition(sep)
            return head.strip(), tail.strip()
    return "", label.strip()


def _same_question(a: str, b: str) -> bool:
    """Do these two labels name the same accused on the same charge?"""
    a_who, a_what = _split_question(a)
    b_who, b_what = _split_question(b)
    if _similarity(a_who, b_who) < _SAME_ACCUSED:
        return False
    # The statute number is the unambiguous identifier when both sides cite one —
    # far more reliable than prose similarity, since "fraud (s.380)" and
    # "possession of proceeds (s.354)" share most of their boilerplate wording.
    a_s, b_s = _STATUTE.findall(a_what), _STATUTE.findall(b_what)
    if a_s and b_s:
        return a_s[0] == b_s[0]
    return _similarity(a_what, b_what) >= _SAME_CHARGE

# The transcript writes each verdict as a bolded bullet, e.g.
#   - **Dana Marlowe — Fraud over $5,000 (s.380(1)(a)): Not Guilty**
# and single-verdict trials as a "**Jury Foreperson:** Verdict: X" line.
_VERDICT_BULLET = re.compile(r"^- \*\*(?P<what>.+?): (?P<outcome>[^:]+?)\*\*\s*$")
_FOREPERSON = re.compile(r"^\*\*Jury Foreperson:\*\* Verdict: (?P<outcome>.+?)\s*$")
_TALLY = re.compile(r"^- Tally: (?P<tally>.+?)\s*$")
_BALLOTS = re.compile(r"^- Ballots counted: (?P<counted>\d+)/(?P<expected>\d+)")
_AGREEMENT = re.compile(r"^- Mean juror self-agreement: (?P<pct>\d+)%")
_MODEL = re.compile(r"^- Model: `(?P<model>[^`]+)`")


def _parse(path: Path) -> dict:
    """Pull the verdicts (and a little provenance) out of one transcript."""
    out: dict = {"file": path.name, "verdicts": {}, "model": "", "ballots": "", "agreement": ""}
    verdict_section = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("## "):
            verdict_section = line.strip() == "## The Verdict"
        if m := _MODEL.match(line):
            out["model"] = m["model"]
        if m := _BALLOTS.match(line):
            out["ballots"] = f"{m['counted']}/{m['expected']}"
        if m := _AGREEMENT.match(line):
            out["agreement"] = f"{m['pct']}%"
        if not verdict_section:
            continue
        if m := _VERDICT_BULLET.match(line):
            out["verdicts"][m["what"].strip()] = m["outcome"].strip()
        elif m := _FOREPERSON.match(line):
            out["verdicts"].setdefault("verdict", m["outcome"].strip())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compare verdicts across saved transcripts.")
    p.add_argument("transcripts", nargs="+", help="Paths to trial_*.md transcripts.")
    args = p.parse_args()

    paths = [Path(t) for t in args.transcripts]
    missing = [p_ for p_ in paths if not p_.is_file()]
    if missing:
        print("Not found: " + ", ".join(str(m) for m in missing))
        return 1
    runs = [_parse(p_) for p_ in paths]

    print(f"# Verdict comparison across {len(runs)} transcript(s)\n")
    for i, r in enumerate(runs, 1):
        extra = " · ".join(x for x in (r["model"], f"ballots {r['ballots']}" if r["ballots"] else "",
                                       f"self-agreement {r['agreement']}" if r["agreement"] else "") if x)
        print(f"Run {i}: {r['file']}" + (f"  ({extra})" if extra else ""))
    print()

    # Build the canonical question list, folding in each run's wording variants.
    keys: list[str] = []
    for r in runs:
        r["aligned"] = {}
        for label, outcome in r["verdicts"].items():
            match = next((k for k in keys if _same_question(k, label)), None)
            if match is None:
                keys.append(label)
                match = label
            r["aligned"][match] = outcome
    if not keys:
        print("No verdicts found — are these completed transcripts?")
        return 1

    width = min(max(len(k) for k in keys), 78)
    header = "Question".ljust(width) + " | " + " | ".join(f"Run {i+1}".ljust(14) for i in range(len(runs)))
    print(header)
    print("-" * len(header))
    stable = 0
    for k in keys:
        vals = [r["aligned"].get(k, "—") for r in runs]
        agreed = len(set(vals)) == 1 and vals[0] != "—"
        stable += agreed
        mark = "" if agreed else "   <-- DIVERGED"
        print(k[:width].ljust(width) + " | " + " | ".join(v.ljust(14) for v in vals) + mark)
    print()
    print(f"{stable}/{len(keys)} questions returned the same verdict in every run "
          f"({round(100 * stable / len(keys))}% agreement).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"use strict";

const form = document.getElementById("caseForm");
const startBtn = document.getElementById("startBtn");
const transcript = document.getElementById("transcript");
const statusEl = document.getElementById("status");
const phasePill = document.getElementById("phase-pill");
const courtSub = document.getElementById("court-sub");

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let stick = true; // auto-scroll only when the user is near the bottom

transcript.addEventListener("scroll", () => {
  stick = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 90;
});

/* ---------------- icons (inline SVG, Lucide-ish) ---------------- */
const ICONS = {
  gavel: '<path d="m13 10-7.4 7.4a2.1 2.1 0 0 1-3-3L10 7"/><path d="m16 13 5.5-5.5"/><path d="m9 6 6 6"/><path d="m13.5 3.5 7 7"/><path d="M5 21h9"/>',
  crown: '<path d="M3 7.5l4 3.5 5-7 5 7 4-3.5L18 19H6L3 7.5z"/><path d="M6 19h12"/>',
  shield: '<path d="M12 3l7 3v5c0 4.5-3 7.6-7 9-4-1.4-7-4.5-7-9V6l7-3z"/>',
  user: '<circle cx="12" cy="8" r="3.2"/><path d="M5.5 20a6.5 6.5 0 0 1 13 0"/>',
  users: '<circle cx="9" cy="8" r="3"/><path d="M2.5 19a6.5 6.5 0 0 1 13 0"/><path d="M16 5.6a3 3 0 0 1 0 5.8"/><path d="M21.5 19a6 6 0 0 0-5-5.9"/>',
  scales: '<path d="M12 3v18"/><path d="M7 21h10"/><path d="M5 7h14"/><path d="M12 4a1.4 1.4 0 1 0 0 2.8A1.4 1.4 0 0 0 12 4Z"/><path d="M5 7l-2.5 6a2.75 2.75 0 0 0 5 0L5 7Z"/><path d="M19 7l-2.5 6a2.75 2.75 0 0 0 5 0L19 7Z"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h5"/>',
  alert: '<path d="M12 3 2.5 20h19L12 3z"/><path d="M12 10v4M12 17h.01"/>',
  dot: '<circle cx="12" cy="12" r="2.6"/>',
};
function svg(name, sw) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${sw || 1.6}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.dot}</svg>`;
}

/* ---------------- roles ---------------- */
function roleInfo(speaker) {
  const s = (speaker || "").toLowerCase();
  if (s.startsWith("judge")) return { key: "judge", tag: "The Bench", icon: "gavel" };
  if (s.startsWith("crown")) return { key: "crown", tag: "Prosecution", icon: "crown" };
  if (s.startsWith("plaintiff") || s.startsWith("commission")) return { key: "crown", tag: "Plaintiff", icon: "crown" };
  if (s.startsWith("defense")) return { key: "defense", tag: "Defense", icon: "shield" };
  if (s.startsWith("witness")) return { key: "witness", tag: "Witness", icon: "user" };
  if (s.startsWith("juror")) return { key: "juror", tag: "Juror", icon: "user" };
  if (s.startsWith("jury")) return { key: "juror", tag: "The Jury", icon: "scales" };
  if (s.startsWith("fact-check")) return { key: "factcheck", tag: "Fact-Check", icon: "alert" };
  if (s.includes("clerk")) return { key: "clerk", tag: "Court Clerk", icon: "file" };
  return { key: "system", tag: "", icon: "dot" };
}

/* ---------------- submit / stream ---------------- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const payload = {
    title: fd.get("title"),
    case_type: fd.get("case_type"),
    jurisdiction: fd.get("jurisdiction"),
    charge_or_claim: fd.get("charge_or_claim"),
    your_side: fd.get("your_side"),
    jury_size: parseInt(fd.get("jury_size"), 10) || 3,
    argument_rounds: parseInt(fd.get("argument_rounds"), 10) || 2,
    deliberation_rounds: parseInt(fd.get("deliberation_rounds"), 10) || 2,
    deliberation_style: fd.get("deliberation_style") || "dialogue",
    model: (fd.get("model") || "").trim() || null,
  };
  const defendants = parseDefendants(fd.get("defendants_raw"));
  const witnesses = parseWitnesses(fd.get("witnesses_raw"));
  if (defendants.length) payload.defendants = defendants;
  if (witnesses.length) payload.witnesses = witnesses;
  await runTrial(payload);
});

// Optional co-accused: one per line, "Name | role | their side".
function parseDefendants(raw) {
  return (raw || "").split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const [name, role, account] = l.split("|").map((s) => (s || "").trim());
    return name ? { name, role: role || "", account: account || "" } : null;
  }).filter(Boolean);
}

const WITNESS_ROLES = ["complainant", "investigator", "expert", "character", "defense_witness", "other"];
// Optional witnesses: one per line, "Name | role | prosecution/defense | what they know".
function parseWitnesses(raw) {
  return (raw || "").split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const [name, role, calledBy, know] = l.split("|").map((s) => (s || "").trim());
    if (!name) return null;
    const r = WITNESS_ROLES.includes((role || "").toLowerCase()) ? role.toLowerCase() : "other";
    const cb = /defen/i.test(calledBy || "") ? "defense" : "prosecution";
    return { name, role: r, called_by: cb, what_they_know: know || "" };
  }).filter(Boolean);
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

async function runTrial(payload) {
  transcript.innerHTML = "";
  stick = true;
  startBtn.disabled = true;
  setStatus("In session", "is-running");
  courtSub.textContent = "Proceedings under way…";
  phasePill.hidden = false;
  phasePill.textContent = "Opening";

  let res;
  try {
    res = await fetch("/api/trial", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    renderError("Could not reach the server: " + err.message);
    return finish("is-error", "Connection lost", "The court could not convene.");
  }
  if (!res.ok) {
    renderError("Server returned status " + res.status);
    return finish("is-error", "Error", "The court could not convene.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const line = buffer.slice(0, idx).replace(/^data:\s?/, "").trim();
      buffer = buffer.slice(idx + 2);
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line); } catch { continue; }
      handleEvent(ev);
    }
  }
  // If the stream closed without the server sending a 'done' event (e.g. server
  // crash or network drop), the button stays disabled and status stays "In session".
  // Re-enable the UI so the user can retry.
  if (startBtn.disabled) finish("is-error", "Disconnected", "The connection was lost.");
}

function finish(cls, statusText, subText) {
  startBtn.disabled = false;
  setStatus(statusText, cls);
  if (subText) courtSub.textContent = subText;
  document.querySelectorAll(".body.typing").forEach((b) => b.classList.remove("typing"));
}

function handleEvent(ev) {
  switch (ev.kind) {
    case "phase":
      phasePill.hidden = false;
      phasePill.textContent = ev.content;
      addPhase(ev.content);
      break;
    case "message":
      addStatement(ev.speaker, ev.content);
      break;
    case "structured":
      renderStructured(ev);
      break;
    case "error":
      renderError(ev.content);
      break;
    case "done":
      finish("is-done", "Adjourned", "Court is adjourned.");
      phasePill.hidden = true;
      break;
  }
}

/* ---------------- rendering ---------------- */
function append(node) {
  transcript.appendChild(node);
  if (stick) transcript.scrollTop = transcript.scrollHeight;
}
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

function addPhase(text) {
  const n = el("div", "phase",
    `<span class="diamond">◆</span>${escapeHtml(text)}<span class="diamond">◆</span>`);
  append(n);
}

function roleCard(speaker) {
  const r = roleInfo(speaker);
  const name = r.key === "juror" && speaker.includes("—")
    ? speaker.split("—")[1].trim()
    : speaker;
  const row = el("div", `card-row r-${r.key}`);
  row.innerHTML =
    `<span class="avatar r-${r.key}">${svg(r.icon)}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">${escapeHtml(name)}</span>` +
      (r.tag ? `<span class="role-tag">${escapeHtml(r.tag)}</span>` : "") +
      `</div><div class="body"></div></div>`;
  return row;
}

function addStatement(speaker, text) {
  const row = roleCard(speaker);
  append(row);
  typewriter(row.querySelector(".body"), text);
}

function typewriter(elBody, text) {
  if (REDUCED) { elBody.textContent = text; return; }
  elBody.classList.add("typing");
  let i = 0;
  const step = Math.max(2, Math.round(text.length / 240));
  const timer = setInterval(() => {
    i += step + Math.floor(Math.random() * 3);
    elBody.textContent = text.slice(0, i);
    if (stick) transcript.scrollTop = transcript.scrollHeight;
    if (i >= text.length) {
      elBody.textContent = text;
      elBody.classList.remove("typing");
      clearInterval(timer);
    }
  }, 16);
}

// Fact-check flags on a statement (anti-hallucination) — an amber warning card.
function renderGrounding(d) {
  const items = (d.flags || []).map((f) =>
    `<li class="fc-flag fc-${escapeHtml(f.severity || "minor")}">` +
      `<span class="fc-sev">${escapeHtml((f.severity || "").toUpperCase())}</span>` +
      `<span class="fc-issue">${escapeHtml(f.issue || "")}</span>` +
      `<span class="fc-claim">${escapeHtml(f.claim || "")}</span>` +
      (f.explanation ? `<div class="fc-exp">${escapeHtml(f.explanation)}</div>` : "") +
    `</li>`
  ).join("");
  const row = el("div", "card-row r-factcheck");
  row.innerHTML =
    `<span class="avatar r-factcheck">${svg("alert")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">Fact-Check</span>` +
        `<span class="role-tag">${(d.flags || []).length} flag(s)</span></div>` +
      `<ul class="fc-list">${items}</ul>` +
    `</div>`;
  append(row);
}

function renderError(msg) {
  append(el("div", "error-card",
    `${svg("alert", 1.7)}<div class="body">${escapeHtml(msg)}</div>`));
}

function renderStructured(ev) {
  const d = ev.data || {};
  if (d._manifest) return renderManifest(d);
  if (d._digest) return renderDigest(d);
  if (d._diagnostics) return renderDiagnostics(d);
  if (d._grounding) return renderGrounding(d);
  if (d._cast) return renderCast(d);
  if (d._directed_verdict) return renderDirectedVerdict(d);
  if (d._record) return renderRecord(d.agreed_record || {});
  if (d._straw) return renderStraw(d);
  if (d._movement) return renderMovement(d);
  if (d._strategy) return renderStrategy(ev.speaker, d);
  if (d.charges_or_claims) return renderCase(ev.speaker, d);
  if (d.jurors) return renderJury(ev.speaker, d.jurors);
  if (d.objection_ruling) return renderObjection(d);
  if (d.tally && d.outcome) return renderVerdict(d);
  if (d.sentence_or_remedy !== undefined && d.verdict_acknowledgement !== undefined) return renderRuling(ev.speaker, d);
  if (d.verdict && d.juror_name) return renderVote(ev.speaker, d);
  addStatement(ev.speaker, ev.content || JSON.stringify(d));
}

// The settings this run was produced under — so a transcript can be traced back.
function renderManifest(d) {
  const chip = (label, value) =>
    `<span class="tag">${escapeHtml(label)}: ${escapeHtml(String(value))}</span>`;
  const row = el("div", "card-row r-clerk");
  row.innerHTML =
    `<span class="avatar r-clerk">${svg("file")}</span>` +
    `<div class="bubble case-panel">` +
      `<div class="who"><span class="name">Trial configuration</span>` +
        `<span class="role-tag">${escapeHtml(d.model || "")}</span></div>` +
      `<div class="tags">` +
        chip("jury", d.jury_size) +
        chip("argument rounds", d.argument_rounds) +
        chip("deliberation", `${d.deliberation_rounds} × ${d.deliberation_style}`) +
        chip("verdict passes", d.verdict_passes) +
        chip("proof threshold", `${d.proof_threshold}%`) +
        (d.strict_elements ? chip("strict elements", "on") : "") +
        (d.calibrated_proof ? chip("calibrated proof", "on") : "") +
        (d.evidence_digest ? chip("evidence digest", "on") : "") +
        (d.straw_poll ? chip("straw poll", "on") : "") +
        (d.grounding_check ? chip("fact-check", "on") : "") +
      `</div>` +
    `</div>`;
  append(row);
}

// The neutral, element-by-element map of the evidence the jury reasons from.
function renderDigest(d) {
  const bullets = (arr, mark) => (arr || [])
    .map((x) => `<li><b>${mark}</b> ${escapeHtml(x)}</li>`).join("");
  const charges = (d.charges || []).map((c) =>
    `<div class="kv" style="margin-top:8px"><b>${escapeHtml(c.charge_label || "")}</b></div>` +
    (c.elements || []).map((e) =>
      `<div class="kv" style="margin-top:4px"><i>${escapeHtml(e.element || "")}</i></div>` +
      `<ul class="facts">${bullets(e.supporting, "✔")}${bullets(e.undermining, "✘")}` +
      `${bullets(e.gaps, "○")}</ul>`
    ).join("")
  ).join("");
  const row = el("div", "card-row r-clerk");
  row.innerHTML =
    `<span class="avatar r-clerk">${svg("scales")}</span>` +
    `<div class="bubble case-panel">` +
      `<div class="who"><span class="name">Evidence Digest</span>` +
        `<span class="role-tag">neutral · element by element</span></div>` +
      charges +
      ((d.undisputed || []).length
        ? `<div class="kv" style="margin-top:8px"><b>Undisputed</b></div>` +
          `<ul class="facts">${(d.undisputed).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
        : "") +
      ((d.disputed || []).length
        ? `<div class="kv" style="margin-top:6px"><b>In dispute</b></div>` +
          `<ul class="facts">${(d.disputed).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
        : "") +
    `</div>`;
  append(row);
}

// How trustworthy the ballots behind this verdict actually are.
function renderDiagnostics(d) {
  const gaps = (d.unmatched_entries || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  const abst = (d.abstentions || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  const row = el("div", "card-row r-juror");
  row.innerHTML =
    `<span class="avatar r-juror">${svg("scales")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">Ballot integrity</span></div>` +
      `<div class="kv"><b>Counted:</b> ${d.ballots_counted}/${d.ballots_expected}` +
        (d.mean_sample_agreement == null
          ? ` · single-sample ballots (not resampled)`
          : ` · <b>mean self-agreement:</b> ${Math.round(d.mean_sample_agreement * 100)}%` +
            ` over ${d.verdict_passes} samples`) +
        `</div>` +
      (abst ? `<div class="kv" style="margin-top:6px"><b>Abstained — excluded from the count</b></div><ul class="facts">${abst}</ul>` : "") +
      (gaps ? `<div class="kv" style="margin-top:6px"><b>Entries that fell back to a top-level vote</b></div><ul class="facts">${gaps}</ul>` : "") +
    `</div>`;
  append(row);
}

// The immutable Agreed Record — the trial's source of truth, shown after intake.
function renderRecord(rec) {
  const list = (label, arr) => (Array.isArray(arr) && arr.length)
    ? `<div class="kv" style="margin-top:6px"><b>${escapeHtml(label)}</b></div>` +
      `<ul class="facts">${arr.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
    : "";
  const row = el("div", "card-row r-clerk");
  row.innerHTML =
    `<span class="avatar r-clerk">${svg("file")}</span>` +
    `<div class="bubble case-panel">` +
      `<div class="who"><span class="name">The Agreed Record</span>` +
        `<span class="role-tag">Source of truth</span></div>` +
      list("Parties", rec.parties) +
      list("Key figures", rec.figures) +
      list("Key dates", rec.dates) +
      list("Admissible facts", rec.admissible_facts) +
      list("Authorities on record", (rec.authorities && rec.authorities.length) ? rec.authorities : ["none"]) +
    `</div>`;
  append(row);
}

function renderCase(speaker, d) {
  const row = el("div", "card-row r-clerk");
  const tags = (d.charges_or_claims || []).map((c) => `<span class="tag">${escapeHtml(c)}</span>`).join("");
  const facts = (d.key_facts || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("");
  row.innerHTML =
    `<span class="avatar r-clerk">${svg("file")}</span>` +
    `<div class="bubble case-panel">` +
      `<div class="who"><span class="name">Case Filed</span><span class="role-tag">${escapeHtml(d.case_caption || "")}</span></div>` +
      (tags ? `<div class="tags">${tags}</div>` : "") +
      (d.summary ? `<div class="kv" style="margin-top:6px">${escapeHtml(d.summary)}</div>` : "") +
      (facts ? `<div class="kv" style="margin-top:8px"><b>Key facts</b></div><ul class="facts">${facts}</ul>` : "") +
    `</div>`;
  append(row);
}

// The auto-cast personalities for the non-jury roles (counsel, bench, witnesses).
function renderCast(d) {
  const chip = (label, p, key) =>
    (p && (p.name || p.style))
      ? `<div class="juror-chip"><span class="mini r-${key}">${svg("user")}</span>` +
        `<div><b>${escapeHtml(label)} · ${escapeHtml(p.name || "")}</b>` +
        `<span>${escapeHtml(p.background || "")}${p.style ? " · " + escapeHtml(p.style) : ""}</span></div></div>`
      : "";
  const wits = (d.witnesses || []).map((w) =>
    `<div class="juror-chip"><span class="mini r-witness">${svg("user")}</span>` +
    `<div><b>${escapeHtml(w.name || "Witness")}</b><span>${escapeHtml(w.style || "")}</span></div></div>`
  ).join("");
  const row = el("div", "card-row r-clerk");
  row.innerHTML =
    `<span class="avatar r-clerk">${svg("users")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">The Cast</span><span class="role-tag">counsel &amp; bench</span></div>` +
      `<div class="jurybox">` +
        chip("Crown", d.crown, "crown") + chip("Defence", d.defense, "defense") + chip("Judge", d.judge, "judge") +
      `</div>` +
      (wits ? `<div class="kv" style="margin-top:8px"><b>Witnesses</b></div><div class="jurybox">${wits}</div>` : "") +
    `</div>`;
  append(row);
}

// A counsel's pre-trial case theory (persistent strategy + planned rebuttal).
function renderStrategy(speaker, d) {
  const r = roleInfo(speaker);
  const pts = (d.strongest_points || []).map((p) => `<li>${escapeHtml(p)}</li>`).join("");
  const row = el("div", `card-row r-${r.key}`);
  row.innerHTML =
    `<span class="avatar r-${r.key}">${svg(r.icon)}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">${escapeHtml(speaker)}</span>` +
        `<span class="role-tag">Case theory</span></div>` +
      (d.theory ? `<div class="body">${escapeHtml(d.theory)}</div>` : "") +
      (pts ? `<div class="kv" style="margin-top:6px"><b>Strongest points</b></div><ul class="facts">${pts}</ul>` : "") +
      (d.opponents_best_point ? `<div class="kv" style="margin-top:6px"><b>Opponent's best point:</b> ${escapeHtml(d.opponents_best_point)}</div>` : "") +
      (d.rebuttal ? `<div class="kv"><b>Planned rebuttal:</b> ${escapeHtml(d.rebuttal)}</div>` : "") +
    `</div>`;
  append(row);
}

function renderJury(speaker, jurors) {
  const items = jurors.map((j) =>
    `<div class="juror-chip"><span class="mini">${svg("user")}</span>` +
    `<div><b>${escapeHtml(j.name || "Juror")}</b>` +
    `<span>${escapeHtml(j.background || "")}${j.disposition ? " · " + escapeHtml(j.disposition) : ""}</span></div></div>`
  ).join("");
  const row = el("div", "card-row r-juror");
  row.innerHTML =
    `<span class="avatar r-juror">${svg("users")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">Jury Empanelled</span><span class="role-tag">${jurors.length} jurors</span></div>` +
      `<div class="jurybox">${items}</div>` +
    `</div>`;
  append(row);
}

// Prefer the structured convict/acquit signal; fall back to careful text (\bnot\b
// avoids matching "cannot"). Returns true when the vote favours the defendant.
function voteIsClear(vote, verdict) {
  return vote ? vote === "acquit" : /\bnot\b|acquit|innocent/i.test(verdict || "");
}

// A juror's per-element findings, shown as a ✓/✗ checklist under their reasoning.
function elementList(findings) {
  if (!Array.isArray(findings) || !findings.length) return "";
  const items = findings.map((ef) => {
    // The juror's own percentage, when they gave one — it is what the standard of
    // proof is enforced against, so it belongs next to the finding.
    const pct = Number.isInteger(ef.probability) ? ` <span class="conf">${ef.probability}%</span>` : "";
    return `<li class="el ${ef.proven ? "el-ok" : "el-no"}">` +
      `<span class="el-mark">${ef.proven ? "✓" : "✗"}</span>` +
      `<span>${escapeHtml(ef.element || "")}${pct}</span></li>`;
  }).join("");
  return `<ul class="elements">${items}</ul>`;
}

// A juror's per-charge votes (multi-charge), each a badge + reasoning + elements.
function chargeVotesBlock(cvs) {
  if (!Array.isArray(cvs) || !cvs.length) return "";
  return cvs.map((cv) =>
    `<div class="cv"><b>${escapeHtml(cv.charge_label || "")}:</b> ` +
    `<span class="vote-badge ${voteIsClear(cv.vote, cv.verdict) ? "vote-clear" : "vote-guilty"}">${escapeHtml(cv.verdict)}</span> ` +
    `${escapeHtml(cv.reasoning || "")}${elementList(cv.element_findings)}</div>`
  ).join("");
}

function renderVote(speaker, d) {
  const name = speaker.includes("—") ? speaker.split("—")[1].trim() : (d.juror_name || "Juror");
  const row = el("div", "card-row r-juror");
  let badges, body;
  if (Array.isArray(d.defendant_votes) && d.defendant_votes.length) {
    // Multi-defendant: one badge per co-accused; per-charge breakdown if present.
    badges = d.defendant_votes.map((dv) =>
      `<span class="vote-badge ${voteIsClear(dv.vote, dv.verdict) ? "vote-clear" : "vote-guilty"}">` +
      `${escapeHtml(dv.defendant_name)}: ${escapeHtml(dv.verdict)}</span>`
    ).join(" ");
    body = d.defendant_votes.map((dv) =>
      `<div class="dv"><b>${escapeHtml(dv.defendant_name)}.</b> ` +
      ((Array.isArray(dv.charge_votes) && dv.charge_votes.length)
        ? chargeVotesBlock(dv.charge_votes)
        : `${escapeHtml(dv.reasoning || "")}${elementList(dv.element_findings)}`) +
      `</div>`
    ).join("");
  } else if (Array.isArray(d.charge_votes) && d.charge_votes.length) {
    // Single accused, multiple charges.
    badges = d.charge_votes.map((cv) =>
      `<span class="vote-badge ${voteIsClear(cv.vote, cv.verdict) ? "vote-clear" : "vote-guilty"}">` +
      `${escapeHtml(cv.charge_label)}: ${escapeHtml(cv.verdict)}</span>`
    ).join(" ");
    body = chargeVotesBlock(d.charge_votes);
  } else {
    badges =
      `<span class="vote-badge ${voteIsClear(d.vote, d.verdict) ? "vote-clear" : "vote-guilty"}">${escapeHtml(d.verdict)}</span>` +
      `<span class="conf">${d.confidence}/10</span>`;
    body = escapeHtml(d.reasoning || "") + elementList(d.element_findings);
  }
  row.innerHTML =
    `<span class="avatar r-juror">${svg("user")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">${escapeHtml(name)}</span>${badges}</div>` +
      `<div class="body">${body}</div>` +
    `</div>`;
  append(row);
}

// The judge's ruling on a defence directed-verdict (no-evidence) motion.
function renderDirectedVerdict(d) {
  const granted = !!d.granted;
  const row = el("div", "card-row r-judge");
  row.innerHTML =
    `<span class="avatar r-judge">${svg("gavel")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">The Bench</span>` +
        `<span class="vote-badge ${granted ? "vote-clear" : "vote-guilty"}">` +
        `${granted ? "DIRECTED VERDICT" : "MOTION DISMISSED"}</span></div>` +
      (d.reasoning ? `<div class="body">${escapeHtml(d.reasoning)}</div>` : "") +
      ((Array.isArray(d.per_defendant) && d.per_defendant.length)
        ? `<div class="kv" style="margin-top:6px"><b>Acquitted:</b> ${escapeHtml(d.per_defendant.join("; "))}</div>`
        : "") +
    `</div>`;
  append(row);
}

function renderObjection(d) {
  const sustained = String(d.objection_ruling || "").toLowerCase().startsWith("sustain");
  const row = el("div", "card-row r-judge");
  row.innerHTML =
    `<span class="avatar r-judge">${svg("gavel")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">The Bench</span>` +
        `<span class="vote-badge ${sustained ? "vote-guilty" : "vote-clear"}">` +
        `${escapeHtml(String(d.objection_ruling || "ruling").toUpperCase())}</span></div>` +
      (d.text ? `<div class="body">${escapeHtml(d.text)}</div>` : "") +
    `</div>`;
  append(row);
}

// The private straw poll taken before discussion (reuses the verdict tally look).
function renderStraw(d) {
  const cols = Object.entries(d.tally || {}).map(([k, n]) =>
    `<div class="col"><div class="num">${n}</div><div class="lbl">${escapeHtml(k)}</div></div>`
  ).join("");
  const n = el("div", "spotlight");
  n.innerHTML =
    `<div class="spot-head">${svg("scales", 1.7)} Straw Poll · before discussion</div>` +
    `<div class="tally">${cols}</div>`;
  append(n);
}

// How the room moved between the straw poll and the final vote.
function renderMovement(d) {
  const line = (t) => Object.entries(t || {}).map(([k, n]) => `${escapeHtml(k)}: ${n}`).join(" · ");
  const flips = (d.flips || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  const row = el("div", "card-row r-juror");
  row.innerHTML =
    `<span class="avatar r-juror">${svg("users")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">How the room moved</span></div>` +
      `<div class="kv"><b>Straw:</b> ${line(d.initial_tally)} &nbsp;→&nbsp; <b>Final:</b> ${line(d.final_tally)}</div>` +
      (flips
        ? `<div class="kv" style="margin-top:6px"><b>Jurors who moved</b></div><ul class="facts">${flips}</ul>`
        : `<div class="kv" style="margin-top:6px">No jurors changed their vote.</div>`) +
    `</div>`;
  append(row);
}

function renderVerdict(d) {
  const cols = Object.entries(d.tally || {}).map(([k, n]) => {
    const win = String(k).toLowerCase() === String(d.outcome).toLowerCase();
    return `<div class="col ${win ? "win" : ""}"><div class="num">${n}</div><div class="lbl">${escapeHtml(k)}</div></div>`;
  }).join("");
  const n = el("div", "spotlight");
  let title = "The Verdict";
  if (d.charge_label && d.defendant_name) title = `${escapeHtml(d.defendant_name)} · ${escapeHtml(d.charge_label)}`;
  else if (d.charge_label) title = `The Verdict — ${escapeHtml(d.charge_label)}`;
  else if (d.defendant_name) title = `The Verdict — ${escapeHtml(d.defendant_name)}`;
  n.innerHTML =
    `<div class="spot-head">${svg("scales", 1.7)} ${title}</div>` +
    `<div class="verdict-outcome">${escapeHtml(d.outcome)}</div>` +
    `<div class="tally">${cols}</div>` +
    (d.dissent_summary ? `<div class="dissent">${escapeHtml(d.dissent_summary)}</div>` : "");
  append(n);
}

function renderRuling(speaker, d) {
  const block = (cls, k, v) => v
    ? `<div class="ruling-block ${cls}">${k ? `<div class="rk">${k}</div>` : ""}<div class="rv">${escapeHtml(v)}</div></div>`
    : "";
  const listBlock = (k, arr) => (Array.isArray(arr) && arr.length)
    ? `<div class="ruling-block"><div class="rk">${k}</div>` +
      `<ul class="facts">${arr.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul></div>`
    : "";
  const n = el("div", "spotlight");
  n.innerHTML =
    `<div class="spot-head">${svg("gavel", 1.7)} The Bench Rules</div>` +
    `<div class="ruling-grid">` +
      block("", "Verdict acknowledged", d.verdict_acknowledgement) +
      block("", "Reasoning", d.reasoning) +
      block("sentence", "Sentence / Remedy", d.sentence_or_remedy) +
      listBlock("Aggravating factors", d.aggravating_factors) +
      listBlock("Mitigating factors", d.mitigating_factors) +
      block("", "Sentencing range", d.sentencing_range) +
      block("", "Restitution", d.restitution) +
      listBlock("Conditions", d.conditions) +
      block("remarks", "", d.closing_remarks) +
    `</div>`;
  append(n);
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

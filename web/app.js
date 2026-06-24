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
  if (s.startsWith("juror")) return { key: "juror", tag: "Juror", icon: "user" };
  if (s.startsWith("jury")) return { key: "juror", tag: "The Jury", icon: "scales" };
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
    jury_size: parseInt(fd.get("jury_size"), 10),
    argument_rounds: parseInt(fd.get("argument_rounds"), 10),
    model: (fd.get("model") || "").trim() || null,
  };
  await runTrial(payload);
});

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

function renderError(msg) {
  append(el("div", "error-card",
    `${svg("alert", 1.7)}<div class="body">${escapeHtml(msg)}</div>`));
}

function renderStructured(ev) {
  const d = ev.data || {};
  if (d.charges_or_claims) return renderCase(ev.speaker, d);
  if (d.jurors) return renderJury(ev.speaker, d.jurors);
  if (d.tally && d.outcome) return renderVerdict(d);
  if (d.sentence_or_remedy !== undefined && d.verdict_acknowledgement !== undefined) return renderRuling(ev.speaker, d);
  if (d.verdict && d.juror_name) return renderVote(ev.speaker, d);
  addStatement(ev.speaker, ev.content || JSON.stringify(d));
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

function renderVote(speaker, d) {
  const clear = /not|acquit|innocent/i.test(d.verdict);
  const name = speaker.includes("—") ? speaker.split("—")[1].trim() : (d.juror_name || "Juror");
  const row = el("div", "card-row r-juror");
  row.innerHTML =
    `<span class="avatar r-juror">${svg("user")}</span>` +
    `<div class="bubble">` +
      `<div class="who"><span class="name">${escapeHtml(name)}</span>` +
        `<span class="vote-badge ${clear ? "vote-clear" : "vote-guilty"}">${escapeHtml(d.verdict)}</span>` +
        `<span class="conf">${d.confidence}/10</span>` +
      `</div><div class="body">${escapeHtml(d.reasoning || "")}</div>` +
    `</div>`;
  append(row);
}

function renderVerdict(d) {
  const cols = Object.entries(d.tally || {}).map(([k, n]) => {
    const win = String(k).toLowerCase() === String(d.outcome).toLowerCase();
    return `<div class="col ${win ? "win" : ""}"><div class="num">${n}</div><div class="lbl">${escapeHtml(k)}</div></div>`;
  }).join("");
  const n = el("div", "spotlight");
  n.innerHTML =
    `<div class="spot-head">${svg("scales", 1.7)} The Verdict</div>` +
    `<div class="verdict-outcome">${escapeHtml(d.outcome)}</div>` +
    `<div class="tally">${cols}</div>` +
    (d.dissent_summary ? `<div class="dissent">${escapeHtml(d.dissent_summary)}</div>` : "");
  append(n);
}

function renderRuling(speaker, d) {
  const block = (cls, k, v) => v
    ? `<div class="ruling-block ${cls}">${k ? `<div class="rk">${k}</div>` : ""}<div class="rv">${escapeHtml(v)}</div></div>`
    : "";
  const n = el("div", "spotlight");
  n.innerHTML =
    `<div class="spot-head">${svg("gavel", 1.7)} The Bench Rules</div>` +
    `<div class="ruling-grid">` +
      block("", "Verdict acknowledged", d.verdict_acknowledgement) +
      block("", "Reasoning", d.reasoning) +
      block("sentence", "Sentence / Remedy", d.sentence_or_remedy) +
      block("remarks", "", d.closing_remarks) +
    `</div>`;
  append(n);
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

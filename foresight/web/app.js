"use strict";

// ---------- helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
  if (opts.body !== undefined && !(opts.body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  } else if (opts.body) {
    init.body = opts.body;
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return opts.raw ? r : r.json();
}

function gb(bytes) {
  if (bytes == null) return "?";
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}
function clock(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function dayLabel(ts) {
  const d = new Date(ts * 1000), today = new Date();
  const y = new Date(); y.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === y.toDateString()) return "Yesterday";
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: d.getFullYear() === today.getFullYear() ? undefined : "numeric" });
}
const timeOf = (ts) => new Date(ts * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
const minutes = (m) => (m.ended_at ? Math.max(1, Math.round((m.ended_at - m.started_at) / 60)) : null);

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), bad ? 7000 : 3500);
}

function ask(title, body, okLabel) {
  const d = $("#dialog");
  $("#dialogTitle").textContent = title;
  $("#dialogBody").textContent = body;
  $("#dialogOk").textContent = okLabel;
  d.returnValue = "";
  d.showModal();
  d.querySelector('[value="cancel"]').focus();
  return new Promise((resolve) => d.addEventListener("close", () => resolve(d.returnValue === "ok"), { once: true }));
}

// ---------- state ----------
const S = {
  st: null,            // /api/state
  meetings: [],
  route: { view: "home" },
  meeting: null,       // full meeting incl. lines
  tab: "mine",
  enhancing: false,
  backlog: {},
  levels: {},
  elapsed: 0,
  sizes: null,
  setupPick: null,
  devices: null,
  ask: { q: "", answer: "", sources: [], busy: false },
  mAsk: { q: "", answer: "", busy: false },
  kbProgress: null,
};

const ready = () => S.st && S.st.setup.runtime_installed && S.st.setup.models.find((m) => m.key === S.st.settings.llm)?.installed;
const recId = () => S.st?.recording?.meeting_id || null;
const whoLabel = (src) => (src === "mic" ? (S.st?.settings.your_name || "You") : src === "system" ? "Others" : "Speaker");

// ---------- routing ----------
function parseHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [view, id] = h.split("/");
  if (view === "meeting" && id) return { view: "meeting", id };
  if (view === "ask" || view === "settings" || view === "setup") return { view };
  return { view: "home" };
}
function go(hash) {
  if (location.hash === hash) route(); else location.hash = hash;
}
async function route() {
  S.route = parseHash();
  if (S.st && !ready() && S.route.view !== "settings") S.route = { view: "setup" };
  if (S.route.view === "meeting") {
    try {
      S.meeting = await api(`/api/meetings/${S.route.id}`);
      S.tab = S.meeting.enhanced ? "enhanced" : "mine";
      S.mAsk = { q: "", answer: "", busy: false };
    } catch (e) {
      toast(`Couldn't open meeting: ${e.message}`, true);
      return go("#/");
    }
  } else {
    S.meeting = null;
  }
  render();
  $("#main").focus({ preventScroll: true });
}

// ---------- rail ----------
function renderRail() {
  const live = recId();
  const btn = $("#recordBtn");
  btn.classList.toggle("live", !!live);
  btn.classList.toggle("primary", true);
  $("#recordLabel").textContent = live ? "Back to recording" : "Start recording";
  btn.disabled = !ready() && !live;
  // home and the live meeting already show the record controls; one primary action per screen
  btn.hidden = S.route.view === "home" || S.route.view === "setup" || (live && S.route.view === "meeting" && S.route.id === live);

  document.querySelectorAll(".nav-item[data-view]").forEach((b) => {
    b.setAttribute("aria-current", S.route.view === b.dataset.view ? "page" : "false");
  });

  const list = $("#meetingList");
  if (!S.meetings.length) {
    list.innerHTML = `<p class="empty-list">Meetings you record or import show up here.</p>`;
    return;
  }
  let html = "", lastDay = "";
  for (const m of S.meetings) {
    const day = dayLabel(m.started_at);
    if (day !== lastDay) { html += `<div class="day">${esc(day)}</div>`; lastDay = day; }
    const cur = S.route.view === "meeting" && S.route.id === m.id;
    let meta;
    if (m.id === live) meta = `<span class="m-live">Recording</span>`;
    else if (m.status === "importing") meta = `<span class="m-live">Transcribing file…</span>`;
    else meta = `${timeOf(m.started_at)}${minutes(m) ? ` · ${minutes(m)} min` : ""}${m.enhanced_len ? " · notes" : ""}`;
    html += `<button type="button" class="m-item" data-id="${esc(m.id)}" aria-current="${cur ? "page" : "false"}">
      <span class="m-title">${esc(m.title)}</span><span class="m-meta">${meta}</span></button>`;
  }
  list.innerHTML = html;
}

// ---------- main ----------
function render() {
  renderRail();
  const v = S.route.view;
  const main = $("#main");
  if (v === "setup") main.innerHTML = setupView();
  else if (v === "meeting" && S.meeting) main.innerHTML = meetingView();
  else if (v === "ask") main.innerHTML = askView();
  else if (v === "settings") main.innerHTML = settingsView();
  else main.innerHTML = homeView();
  afterRender(v);
}

function afterRender(v) {
  if (v === "setup" && !S.sizes) loadSizes();
  if (v === "settings" && !S.devices) loadDevices();
  if (v === "meeting") {
    const tl = $("#tLines");
    if (tl) tl.scrollTop = tl.scrollHeight;
  }
}

// ---------- setup ----------
async function loadSizes() {
  try {
    S.sizes = await api("/api/setup/sizes");
  } catch (e) {
    S.sizes = { error: e.message };
  }
  if (S.route.view === "setup") render();
}

function setupView() {
  const st = S.st.setup;
  const p = st.progress || {};
  if (!S.setupPick) {
    S.setupPick = new Set([S.st.settings.llm, "embeddinggemma-2"]);
  }
  const sizes = S.sizes || {};
  const llmModels = st.models.filter((m) => m.role === "llm");
  const embed = st.models.find((m) => m.role === "embed");
  const why = {
    "gemma-4-e4b": "Transcribes, writes notes and answers questions. Best quality; needs about 9 GB of GPU memory.",
    "gemma-4-e2b": "Same jobs, smaller and faster, slightly less accurate. For GPUs with under 8 GB.",
    "embeddinggemma-2": "Indexes your linked folders and past meetings so you can search and ask across them.",
  };
  let total = st.runtime_installed ? 0 : (sizes.runtime || 0);
  for (const m of st.models) if (S.setupPick.has(m.key) && !m.installed) total += sizes[m.key] || 0;
  const sizesKnown = S.sizes && !S.sizes.error;

  const row = (m, type, name) => `
    <label class="choice">
      <input type="${type}" name="${name}" value="${m.key}" ${S.setupPick.has(m.key) ? "checked" : ""} ${p.running || m.installed ? "disabled" : ""}>
      <span><span class="name">${esc(m.label)}${m.key === "gemma-4-e4b" ? " — recommended" : ""}</span><br><span class="why">${why[m.key] || ""}</span></span>
      <span class="size">${m.installed ? "Installed" : sizesKnown ? gb(sizes[m.key]) : "…"}</span>
    </label>`;

  let progress = "";
  if (p.running && p.steps) {
    const pct = p.total ? Math.min(100, (100 * p.done) / p.total) : 0;
    progress = `<div class="progress" aria-live="polite">
      <div class="row"><span>Step ${p.step} of ${p.steps}: ${esc(p.label || "")}</span><span class="num">${p.total ? `${gb(p.done)} of ${gb(p.total)}` : "connecting…"}</span></div>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct.toFixed(0)}"><i style="width:${pct}%"></i></div>
    </div>`;
  } else if (p.error) {
    progress = `<p class="error progress">Download stopped: ${esc(p.error)}. Already-downloaded parts are kept; press the button to resume.</p>`;
  }

  return `<div class="page">
    <h1 class="hero num">${sizesKnown ? esc(gb(total).split(" ")[0]) : "…"}<span class="unit"> ${sizesKnown ? esc(gb(total).split(" ")[1]) : ""} to download once</span></h1>
    <p class="lede">After this download, transcription, notes and search run on this PC with no internet connection. Audio, transcripts and files never leave it.</p>
    <dl class="facts">
      <div class="fact"><dt>Runs on</dt><dd>${st.backend === "cuda" ? "NVIDIA GPU (CUDA 12.4)" : st.backend === "vulkan" ? "GPU via Vulkan" : "CPU only"}</dd></div>
      <div class="fact"><dt>Engine</dt><dd>llama.cpp ${esc(st.llama_tag)}${st.runtime_installed ? " · installed" : sizesKnown && sizes.runtime ? ` · ${gb(sizes.runtime)}` : ""}</dd></div>
      <div class="fact"><dt>Saved in</dt><dd class="small" style="word-break:break-all">${esc(S.st.data_dir)}</dd></div>
      <div class="fact"><dt>Free space there</dt><dd class="num">${sizesKnown && sizes.free != null ? gb(sizes.free) : "…"}</dd></div>
    </dl>
    ${S.sizes?.error ? `<p class="error">Couldn't reach Hugging Face to look up file sizes: ${esc(S.sizes.error)}</p>` : ""}
    <div role="radiogroup" aria-label="Language model">${llmModels.map((m) => row(m, "radio", "llm")).join("")}</div>
    ${embed ? row(embed, "checkbox", "embed") : ""}
    <div class="actions" style="margin-top:var(--s6)">
      ${p.running
        ? `<button class="btn text" type="button" data-act="cancel-setup">Cancel download</button>`
        : `<button class="btn primary big" type="button" data-act="start-setup" ${sizesKnown ? "" : "disabled"}>Download ${sizesKnown ? gb(total) : ""}</button>`}
    </div>
    ${progress}
  </div>`;
}

// ---------- home ----------
function homeView() {
  const s = S.st.settings, rt = S.st.runtime;
  const llm = S.st.setup.models.find((m) => m.key === s.llm);
  let hero = "Ready to record", lede = "Records your microphone and everything your PC plays, and transcribes both on this computer as the meeting happens.";
  if (rt.llm === "starting") { hero = "Loading Gemma…"; lede = "Loading the model onto the GPU takes 10–30 seconds. You can start recording now; transcription catches up once it's loaded."; }
  if (rt.llm === "error") { hero = "Gemma didn't start"; lede = rt.error || "Check Settings, or the log in the data folder."; }
  const backend = { cuda: "CUDA", vulkan: "Vulkan", cpu: "CPU" }[S.st.setup.backend] || S.st.setup.backend;
  return `<div class="page">
    <h1 class="hero">${esc(hero)}</h1>
    <p class="lede ${rt.llm === "error" ? "error" : ""}" style="white-space:pre-wrap">${esc(lede)}</p>
    <dl class="facts">
      <div class="fact"><dt>Your voice</dt><dd>${s.mic_device == null ? "Default microphone" : `Mic #${s.mic_device}`}</dd></div>
      <div class="fact"><dt>Other people</dt><dd>${s.capture_system_audio ? (s.speaker_device == null ? "Default speakers (loopback)" : `Output #${s.speaker_device}`) : "Not captured"}</dd></div>
      <div class="fact"><dt>Language</dt><dd>${esc(s.language)}</dd></div>
      <div class="fact"><dt>Model</dt><dd>${esc(llm?.label || s.llm)} · ${esc(backend)}</dd></div>
    </dl>
    <div class="actions">
      <button class="btn primary big" type="button" data-act="record"><span class="dot" aria-hidden="true"></span>Start recording</button>
      <button class="btn text" type="button" data-act="import">Transcribe a file…</button>
    </div>
    ${s.capture_system_audio && !s.headphones ? `<p class="small muted" style="margin-top:var(--s5)">Using speakers? Your mic will hear the other side too; matching lines are removed from your channel. With headphones, turn that off in Settings.</p>` : ""}
  </div>`;
}

// ---------- meeting ----------
function mdToHtml(md, mineSet) {
  const norm = (t) => t.toLowerCase().replace(/[*_`#>\[\]()-]/g, " ").replace(/\s+/g, " ").trim();
  const inline = (t) => esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const cls = (t) => {
    if (!mineSet) return "";
    const n = norm(t);
    for (const m of mineSet) if (m && (n === m || n.startsWith(m))) return "mine";
    return "ai";
  };
  let html = "", inList = false;
  for (const raw of md.split("\n")) {
    const line = raw.replace(/\s+$/, "");
    let m;
    if ((m = line.match(/^\s*[-*+]\s+\[( |x|X)\]\s+(.*)$/))) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li class="task ${m[1] !== " " ? "done" : ""} ${cls(m[2])}">${inline(m[2])}</li>`;
    } else if ((m = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/))) {
      if (!inList) { html += "<ul>"; inList = true; }
      const depth = Math.min(4, Math.floor(m[1].length / 2));
      html += `<li class="${cls(m[3])}" style="margin-left:${depth * 1.2}em">${inline(m[3])}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if ((m = line.match(/^(#{1,3})\s+(.*)$/))) html += `<h${m[1].length} class="${cls(m[2])}">${inline(m[2])}</h${m[1].length}>`;
      else if (line.trim()) html += `<p class="${cls(line)}">${inline(line)}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function userLineSet(notes) {
  const set = new Set();
  for (const l of (notes || "").split("\n")) {
    const n = l.toLowerCase().replace(/[*_`#>\[\]()-]/g, " ").replace(/\s+/g, " ").trim();
    if (n.length >= 2) set.add(n);
  }
  return set;
}

function lineHtml(l) {
  return `<div class="t-line ${esc(l.source)}" data-line="${l.id}"><span class="t-time">${clock(l.start)}</span><span><span class="t-who">${esc(whoLabel(l.source))}</span>${esc(l.text)}</span></div>`;
}

function meetingHead(m) {
  const live = m.id === recId();
  if (live) {
    const paused = S.st.recording.paused;
    return `<div class="status">
        <span class="timer num ${paused ? "" : "live"}" id="timer">${clock(S.elapsed || S.st.recording.elapsed)}</span>
        <div class="meters" aria-label="Input levels">
          <div class="meter"><span>${esc(whoLabel("mic"))}</span><span class="track"><i id="lvl-mic"></i></span></div>
          ${S.st.settings.capture_system_audio ? `<div class="meter"><span>Others</span><span class="track"><i id="lvl-system"></i></span></div>` : ""}
        </div>
      </div>
      <div class="actions">
        <button class="btn text" type="button" data-act="pause">${paused ? "Resume" : "Pause"}</button>
        <button class="btn stop" type="button" data-act="stop">Stop</button>
      </div>`;
  }
  const backlog = S.backlog[m.id] || 0;
  const status = m.status === "importing" || backlog ? `Transcribing… ${backlog ? `${backlog} segment${backlog === 1 ? "" : "s"} left` : ""}` : "";
  const when = new Date(m.started_at * 1000).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  return `<div class="status small muted"><span>${esc(when)}${minutes(m) ? ` · ${minutes(m)} min` : ""}</span>${status ? `<span id="mStatus">${esc(status)}</span>` : ""}</div>
    <button class="btn text" type="button" data-act="delete-meeting">Delete</button>`;
}

function meetingView() {
  const m = S.meeting;
  const live = m.id === recId();
  const hasEnh = !!m.enhanced || S.enhancing;
  const tab = hasEnh ? S.tab : "mine";
  const notesBody = tab === "enhanced"
    ? `<div class="md" id="enhanced">${S.enhancing && !m.enhanced ? `<p class="muted">Reading the transcript…</p>` : mdToHtml(m.enhanced, userLineSet(m.notes))}</div>`
    : `<textarea class="notes-input" id="notes" aria-label="Your notes" placeholder="Type short notes while you listen. “- budget?” or “ask Lars re: billing” is enough. Enhance turns them into full notes using the transcript.">${esc(m.notes)}</textarea>`;
  const lines = m.lines.length ? m.lines.map(lineHtml).join("") : `<p class="t-empty">${live ? "Lines appear a few seconds after someone speaks." : "No transcript."}</p>`;
  const enhanceLabel = S.enhancing ? "Enhancing…" : m.enhanced ? "Enhance again" : "Enhance notes";
  return `<div class="meeting">
    <header class="m-head">
      <input class="title-input" id="title" value="${esc(m.title)}" aria-label="Meeting title">
      ${meetingHead(m)}
    </header>
    <div class="m-body">
      <section class="notes-col" aria-label="Notes">
        <div class="tabs" role="tablist">
          ${hasEnh ? `<button class="tab" role="tab" type="button" data-tab="enhanced" aria-selected="${tab === "enhanced"}">Enhanced</button>
          <button class="tab" role="tab" type="button" data-tab="mine" aria-selected="${tab === "mine"}">My notes</button>` : `<span class="small muted">Your notes · saved as you type</span>`}
          <span class="spacer"></span>
          ${tab === "enhanced" && m.enhanced ? `<button class="btn text" type="button" data-act="copy">Copy Markdown</button>` : ""}
          <button class="btn ${live ? "" : "primary"}" type="button" data-act="enhance" ${S.enhancing || !m.lines.length ? "disabled" : ""}>${enhanceLabel}</button>
        </div>
        <div class="sheet">${notesBody}</div>
        ${tab === "enhanced" && m.enhanced ? `<p class="legend"><b>Bold lines</b> are your own notes; the rest came from the transcript.</p>` : ""}
      </section>
      <section class="transcript-col" aria-label="Transcript">
        <div class="t-head"><span>Transcript</span><span class="muted num" id="tCount">${m.lines.length} lines</span></div>
        <div class="t-lines" id="tLines" aria-live="polite">${lines}</div>
        ${S.mAsk.q ? `<div class="answer"><div class="q">${esc(S.mAsk.q)}</div><span id="mAnswer">${esc(S.mAsk.answer || "…")}</span></div>` : ""}
        <form class="askbar" data-form="meeting-ask"><input name="q" placeholder="Ask about this meeting" aria-label="Ask about this meeting" autocomplete="off"><button class="btn" type="submit" ${S.mAsk.busy ? "disabled" : ""}>Ask</button></form>
      </section>
    </div>
  </div>`;
}

// ---------- ask ----------
function askView() {
  const kb = S.st.kb, s = S.st.settings;
  const embedOk = S.st.setup.models.find((m) => m.key === "embeddinggemma-2")?.installed;
  const prog = S.kbProgress && S.kbProgress.total && S.kbProgress.done < S.kbProgress.total
    ? `Indexing ${S.kbProgress.done + 1} of ${S.kbProgress.total}: ${esc(S.kbProgress.current || "")}`
    : `${kb.files.toLocaleString()} files · ${kb.chunks.toLocaleString()} passages indexed`;
  const sources = S.ask.sources.map((x) => `<div class="source">
      <div class="s-head"><span class="num">[${x.n}]</span><span>${x.ref.startsWith("meeting:") ? `<button class="link" type="button" data-open-meeting="${esc(x.ref.slice(8))}">${esc(x.title)}</button>` : `<button class="link" type="button" data-open-file="${esc(x.ref)}">${esc(x.title)}</button>`}</span></div>
      <div class="s-ex">${esc(x.excerpt)}${x.excerpt.length >= 400 ? "…" : ""}</div></div>`).join("");
  return `<div class="page">
    ${embedOk ? "" : `<p class="error">EmbeddingGemma 2 isn't downloaded, so only single meetings can be asked about. <button class="link" type="button" data-act="get-embed">Download it</button></p>`}
    <form class="ask-input" data-form="ask">
      <input name="q" value="${esc(S.ask.q)}" placeholder="What did we decide about the Q3 budget?" aria-label="Question" autocomplete="off" ${embedOk ? "" : "disabled"}>
      <button class="btn primary big" type="submit" ${S.ask.busy || !embedOk ? "disabled" : ""}>Ask</button>
    </form>
    <div class="ask-answer" id="askAnswer" aria-live="polite">${esc(S.ask.answer)}</div>
    <div class="sources">${sources}</div>
    <h2>Linked folders</h2>
    <p class="small muted" id="kbStatus">${prog}. Searches include these folders and every meeting.</p>
    <ul class="folders">${s.folders.map((f, i) => `<li><span>${esc(f)}</span><button class="btn text" type="button" data-remove-folder="${i}">Remove</button></li>`).join("") || `<li class="muted">No folders linked yet.</li>`}</ul>
    <form class="add-folder" data-form="add-folder">
      ${window.pywebview ? "" : `<input name="path" placeholder="C:\\Users\\you\\Documents\\Notes" aria-label="Folder path">`}
      <button class="btn" type="submit">${window.pywebview ? "Link a folder…" : "Link folder"}</button>
      <button class="btn text" type="button" data-act="rescan">Rescan</button>
    </form>
    <p class="small muted">Reads .txt, .md, .pdf, .docx, .csv, .json and .html files up to 25 MB.</p>
  </div>`;
}

// ---------- settings ----------
async function loadDevices() {
  try { S.devices = await api("/api/devices"); } catch { S.devices = []; }
  if (S.route.view === "settings") render();
}

function settingsView() {
  const s = S.st.settings;
  const devs = S.devices || [];
  const opts = (kind, cur) => `<option value="" ${cur == null ? "selected" : ""}>Windows default</option>` +
    devs.filter((d) => d.kind === kind).map((d) => `<option value="${d.index}" ${cur === d.index ? "selected" : ""}>${esc(d.name)}${d.default ? " (default)" : ""}</option>`).join("");
  const llms = S.st.setup.models.filter((m) => m.role === "llm");
  return `<div class="page">
    <h1 class="hero" style="font-size:56px">Settings</h1>
    <div class="form">
      <div class="field"><label for="f-language">Meeting language</label>
        <select id="f-language" data-setting="language">${S.st.languages.map((l) => `<option ${l === s.language ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
        <span class="hint">Gemma transcribes into this language. Pick it before you start recording.</span></div>
      <div class="field"><label for="f-name">Your name in transcripts</label>
        <input type="text" id="f-name" data-setting="your_name" value="${esc(s.your_name)}"></div>
      <div class="field"><label for="f-mic">Microphone</label>
        <select id="f-mic" data-setting="mic_device" data-int>${opts("input", s.mic_device)}</select></div>
      <div class="field"><label for="f-spk">Capture sound from</label>
        <select id="f-spk" data-setting="speaker_device" data-int>${opts("output", s.speaker_device)}</select>
        <span class="hint">Whatever plays on this output (Teams, Zoom, Meet in a browser) is transcribed as “Others”.</span></div>
      <label class="check"><input type="checkbox" data-setting="capture_system_audio" ${s.capture_system_audio ? "checked" : ""}><span><b>Capture system audio</b><br><span class="hint">Off records only your microphone.</span></span></label>
      <label class="check"><input type="checkbox" data-setting="headphones" ${s.headphones ? "checked" : ""}><span><b>I use headphones</b><br><span class="hint">With speakers, your mic picks up the other side; Foresight removes those duplicate lines unless this is on.</span></span></label>
      <label class="check"><input type="checkbox" data-setting="save_audio" ${s.save_audio ? "checked" : ""}><span><b>Keep audio recordings</b><br><span class="hint">FLAC files in the meeting's folder, about 60 MB per hour per channel.</span></span></label>
      <div class="field"><label for="f-llm">Model</label>
        <select id="f-llm" data-setting="llm">${llms.map((m) => `<option value="${m.key}" ${m.key === s.llm ? "selected" : ""} ${m.installed ? "" : "disabled"}>${esc(m.label)}${m.installed ? "" : " (not downloaded)"}</option>`).join("")}</select>
        <span class="hint"><button class="link" type="button" data-act="goto-setup">Download models</button></span></div>
      <div class="field"><label for="f-backend">GPU backend</label>
        <select id="f-backend" data-setting="backend">${[["auto", "Automatic"], ["cuda", "NVIDIA (CUDA)"], ["vulkan", "AMD or Intel (Vulkan)"], ["cpu", "CPU only (slow)"]].map(([v, l]) => `<option value="${v}" ${v === s.backend ? "selected" : ""}>${l}</option>`).join("")}</select>
        <span class="hint">Auto uses CUDA on NVIDIA cards. Changing it downloads that llama.cpp build.</span></div>
      <div class="field"><span class="label">Data folder</span><span class="small" style="word-break:break-all">${esc(S.st.data_dir)}</span>
        <span class="hint">Meetings, audio, models and the search index. Delete it to remove everything.</span></div>
    </div>
  </div>`;
}

// ---------- actions ----------
async function refreshState() {
  S.st = await api("/api/state");
}
async function refreshMeetings() {
  S.meetings = await api("/api/meetings");
  renderRail();
}

async function startRecording() {
  if (recId()) return go(`#/meeting/${recId()}`);
  try {
    const r = await api("/api/recording/start", { body: {} });
    await refreshState();
    await refreshMeetings();
    S.elapsed = 0;
    go(`#/meeting/${r.meeting_id}`);
  } catch (e) {
    toast(e.message, true);
  }
}

let saveTimer;
function saveNotesSoon(value) {
  S.meeting.notes = value;
  clearTimeout(saveTimer);
  const id = S.meeting.id;
  saveTimer = setTimeout(() => api(`/api/meetings/${id}`, { method: "PATCH", body: { notes: value } }).catch((e) => toast(`Notes not saved: ${e.message}`, true)), 500);
}

async function enhance() {
  const m = S.meeting;
  clearTimeout(saveTimer);
  await api(`/api/meetings/${m.id}`, { method: "PATCH", body: { notes: m.notes } });
  S.enhancing = true;
  S.tab = "enhanced";
  const previous = m.enhanced;
  m.enhanced = "";
  render();
  try {
    const r = await api(`/api/meetings/${m.id}/enhance`, { body: {}, raw: true });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      m.enhanced += dec.decode(value, { stream: true });
      const el = $("#enhanced");
      if (el && S.meeting === m) el.innerHTML = mdToHtml(m.enhanced, userLineSet(m.notes));
    }
  } catch (e) {
    m.enhanced = previous;
    toast(e.message, true);
  }
  S.enhancing = false;
  if (S.meeting === m) render();
  refreshMeetings();
}

async function streamAsk(body, onSources, onToken) {
  const r = await api("/api/ask", { body, raw: true });
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const msg = JSON.parse(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
      if (msg.sources) onSources(msg.sources);
      if (msg.t) onToken(msg.t);
      if (msg.error) throw new Error(msg.error);
    }
  }
}

async function saveSetting(key, value) {
  try {
    const s = await api("/api/settings", { body: { [key]: value } });
    S.st.settings = s;
    if (key === "backend") { await refreshState(); if (!ready()) go("#/setup"); }
  } catch (e) {
    toast(e.message, true);
  }
}

document.addEventListener("click", async (ev) => {
  const t = ev.target.closest("button, [data-tab]");
  if (!t) return;
  if (t.id === "recordBtn") return startRecording();
  if (t.classList.contains("nav-item") && t.dataset.view) return go(`#/${t.dataset.view}`);
  if (t.classList.contains("m-item")) { $("#rail").classList.remove("show-list"); return go(`#/meeting/${t.dataset.id}`); }
  if (t.dataset.tab) { S.tab = t.dataset.tab; return render(); }
  if (t.dataset.openMeeting) return go(`#/meeting/${t.dataset.openMeeting}`);
  if (t.dataset.openFile) return api("/api/open", { body: { path: t.dataset.openFile } }).catch((e) => toast(e.message, true));
  if (t.dataset.removeFolder !== undefined) {
    const folders = S.st.settings.folders.filter((_, i) => i !== Number(t.dataset.removeFolder));
    await saveSetting("folders", folders);
    return render();
  }
  const act = t.dataset.act;
  if (!act) return;
  if (act === "record") return startRecording();
  if (act === "import") return $("#importInput").click();
  if (act === "stop") {
    t.disabled = true;
    try { await api("/api/recording/stop", { body: {} }); } catch (e) { toast(e.message, true); }
    await refreshState();
    S.backlog[S.meeting.id] = S.backlog[S.meeting.id] || 0;
    await refreshMeetings();
    return render();
  }
  if (act === "pause") {
    const paused = !S.st.recording.paused;
    await api("/api/recording/pause", { body: { paused } });
    S.st.recording.paused = paused;
    return render();
  }
  if (act === "enhance") return enhance();
  if (act === "copy") {
    await navigator.clipboard.writeText(S.meeting.enhanced);
    return toast("Copied as Markdown");
  }
  if (act === "delete-meeting") {
    if (!(await ask(`Delete “${S.meeting.title}”?`, "Its notes, transcript and audio are removed from this PC. This can't be undone.", "Delete meeting"))) return;
    await api(`/api/meetings/${S.meeting.id}`, { method: "DELETE" });
    await refreshMeetings();
    return go("#/");
  }
  if (act === "start-setup") {
    try {
      await api("/api/setup/install", { body: { keys: [...S.setupPick] } });
      await api("/api/settings", { body: { llm: [...S.setupPick].find((k) => k.startsWith("gemma")) || S.st.settings.llm } });
      await refreshState();
      render();
    } catch (e) { toast(e.message, true); }
    return;
  }
  if (act === "cancel-setup") return api("/api/setup/cancel", { body: {} });
  if (act === "get-embed") { S.setupPick = new Set(["embeddinggemma-2"]); S.route = { view: "setup" }; return render(); }
  if (act === "goto-setup") { S.route = { view: "setup" }; return render(); }
  if (act === "rescan") { await api("/api/kb/scan", { body: {} }); return toast("Rescanning linked folders"); }
  if (act === "toggle-list") { $("#rail").classList.toggle("show-list"); }
});

document.addEventListener("change", async (ev) => {
  const t = ev.target;
  if (t.name === "llm") { S.setupPick.delete("gemma-4-e4b"); S.setupPick.delete("gemma-4-e2b"); S.setupPick.add(t.value); return render(); }
  if (t.name === "embed") { t.checked ? S.setupPick.add(t.value) : S.setupPick.delete(t.value); return render(); }
  if (t.id === "importInput" && t.files[0]) {
    const fd = new FormData();
    fd.append("file", t.files[0]);
    t.value = "";
    toast("Uploading file for transcription…");
    try {
      const r = await api("/api/import", { body: fd });
      await refreshMeetings();
      go(`#/meeting/${r.meeting_id}`);
    } catch (e) { toast(e.message, true); }
    return;
  }
  if (t.dataset.setting) {
    let v = t.type === "checkbox" ? t.checked : t.value;
    if (t.dataset.int !== undefined) v = v === "" ? null : Number(v);
    return saveSetting(t.dataset.setting, v);
  }
  if (t.id === "title") {
    S.meeting.title = t.value.trim() || S.meeting.title;
    await api(`/api/meetings/${S.meeting.id}`, { method: "PATCH", body: { title: S.meeting.title } });
    refreshMeetings();
  }
});

document.addEventListener("input", (ev) => {
  if (ev.target.id === "notes") saveNotesSoon(ev.target.value);
});

document.addEventListener("keydown", (ev) => {
  if (ev.target.id === "title" && ev.key === "Enter") ev.target.blur();
});

document.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const kind = form.dataset.form;
  if (kind === "ask") {
    const q = form.q.value.trim();
    if (!q) return;
    S.ask = { q, answer: "", sources: [], busy: true };
    render();
    try {
      await streamAsk({ question: q },
        (src) => { S.ask.sources = src; },
        (tok) => { S.ask.answer += tok; const el = $("#askAnswer"); if (el) el.textContent = S.ask.answer; });
    } catch (e) { toast(e.message, true); }
    S.ask.busy = false;
    if (S.route.view === "ask") render();
  }
  if (kind === "meeting-ask") {
    const q = form.q.value.trim();
    if (!q) return;
    S.mAsk = { q, answer: "", busy: true };
    render();
    try {
      await streamAsk({ question: q, meeting_id: S.meeting.id }, () => {},
        (tok) => { S.mAsk.answer += tok; const el = $("#mAnswer"); if (el) el.textContent = S.mAsk.answer; });
    } catch (e) { toast(e.message, true); }
    S.mAsk.busy = false;
    if (S.route.view === "meeting") render();
  }
  if (kind === "add-folder") {
    let path = form.path ? form.path.value.trim() : null;
    if (window.pywebview) path = await window.pywebview.api.pick_folder();
    if (!path) return;
    if (S.st.settings.folders.includes(path)) return toast("That folder is already linked");
    await saveSetting("folders", [...S.st.settings.folders, path]);
    render();
  }
});

window.addEventListener("hashchange", route);

// ---------- live events ----------
function connect() {
  const ws = new WebSocket(`ws://${location.host}/api/events`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1000);
}

async function handle(ev) {
  const cur = S.meeting;
  switch (ev.type) {
    case "levels": {
      S.levels = ev.levels; S.elapsed = ev.elapsed;
      const timer = $("#timer");
      if (timer && !S.st.recording?.paused) timer.textContent = clock(ev.elapsed);
      for (const [k, v] of Object.entries(ev.levels)) {
        const el = $(`#lvl-${k}`);
        // perceptual-ish scale: -60 dBFS..0 -> 0..100%
        if (el) el.style.width = `${Math.max(0, Math.min(100, ((20 * Math.log10(v + 1e-6)) + 60) / 60 * 100))}%`;
      }
      break;
    }
    case "line":
      if (cur && cur.id === ev.meeting_id && !cur.lines.some((l) => l.id === ev.line.id)) {
        cur.lines.push(ev.line);
        cur.lines.sort((a, b) => a.start - b.start);
        const box = $("#tLines");
        if (box) {
          const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
          box.querySelector(".t-empty")?.remove();
          const after = [...box.children].find((c) => cur.lines.find((l) => String(l.id) === c.dataset.line)?.start > ev.line.start);
          after ? after.insertAdjacentHTML("beforebegin", lineHtml(ev.line)) : box.insertAdjacentHTML("beforeend", lineHtml(ev.line));
          if (stick) box.scrollTop = box.scrollHeight;
          $("#tCount").textContent = `${cur.lines.length} lines`;
          const btn = document.querySelector('[data-act="enhance"]');
          if (btn && !S.enhancing) btn.disabled = false;
        }
      }
      break;
    case "line_removed":
      if (cur && cur.id === ev.meeting_id) {
        cur.lines = cur.lines.filter((l) => l.id !== ev.id);
        document.querySelector(`[data-line="${ev.id}"]`)?.remove();
      }
      break;
    case "backlog":
      S.backlog[ev.meeting_id] = ev.pending;
      if (cur && cur.id === ev.meeting_id && cur.id !== recId()) {
        const el = $("#mStatus");
        if (el) el.textContent = ev.pending ? `Transcribing… ${ev.pending} segment${ev.pending === 1 ? "" : "s"} left` : "";
      }
      break;
    case "recording":
      await refreshState();
      await refreshMeetings();
      if (cur && cur.id === ev.meeting_id && ev.state === "done") {
        S.meeting = await api(`/api/meetings/${cur.id}`);
        if (!S.enhancing) render();
      } else if (S.route.view === "meeting" && !S.enhancing && document.activeElement?.id !== "notes") {
        render();
      }
      break;
    case "setup":
      S.st.setup.progress = ev;
      if (ev.finished) {
        await refreshState();
        toast("Download complete. Loading the model…");
        S.sizes = null;
        go("#/");
      } else if (S.route.view === "setup") {
        render();
      }
      break;
    case "runtime":
      S.st.runtime = { llm: ev.llm, embed: ev.embed, error: ev.error };
      if (ev.llm === "error") toast(`Gemma didn't start: ${String(ev.error).split("\n")[0]}`, true);
      if (S.route.view === "home") render();
      break;
    case "kb_progress":
      S.kbProgress = ev;
      if (ev.done === ev.total) { await refreshState(); }
      if (S.route.view === "ask") {
        const el = $("#kbStatus");
        if (el) render();
      }
      break;
    case "error":
      toast(`${ev.where === "import" ? "Import" : "Transcription"} error: ${ev.message}`, true);
      break;
  }
}

// ---------- boot ----------
(async function boot() {
  // narrow windows: a toggle for the meeting list
  $(".nav").insertAdjacentHTML("beforeend", `<button type="button" class="nav-item narrow-only" data-act="toggle-list">Meetings</button>`);
  const style = document.createElement("style");
  style.textContent = `.narrow-only{display:none}@media (max-width:760px){.narrow-only{display:block}}`;
  document.head.appendChild(style);
  try {
    await refreshState();
    await refreshMeetings();
  } catch (e) {
    $("#main").innerHTML = `<div class="page"><h1 class="hero">Can't reach the app</h1><p class="lede error">${esc(e.message)}</p></div>`;
    return;
  }
  connect();
  route();
  // the pywebview bridge arrives after load; re-render so native folder picking appears
  window.addEventListener("pywebviewready", () => { if (S.route.view === "ask") render(); });
})();

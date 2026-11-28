/* FilingLens app page: talks to the FilingLens API. No dependencies.
 * Every string from the API is escaped before it is inserted; the only markup added is ours
 * (citation buttons and <mark> highlights). */

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);

/* Plain-English names for the parts of a 10-K. */
const ITEM_NAMES = {
  "1": "What the business does",
  "1A": "Risks",
  "3": "Lawsuits",
  "7": "Management's explanation",
  "7A": "Market risks",
};
const STEP_NAMES = {
  search_filings: "Searched the reports",
  compare_sections: "Compared two years",
  get_financials: "Looked up reported numbers",
  compute_ratio: "Worked out a ratio from reported numbers",
  route: "Decided which tools to use",
  write: "Wrote the answer",
};
const CONFIDENCE = { high: "Strong evidence", medium: "Some evidence", low: "Thin evidence" };

/* One-sentence explanations for the numbers page, keyed by API metric name. */
const METRIC_HELP = {
  revenue: ["Revenue", "All the money coming in from sales."],
  operating_income: ["Operating profit", "What's left from sales after the costs of running the business, before interest and tax."],
  net_income: ["Net profit", "The bottom line: what's left after every cost, interest payment and tax."],
  gross_margin: ["Gross margin", "Of each $1 of sales, how much is left after the direct cost of what was sold."],
  operating_margin: ["Operating margin", "Of each $1 of sales, how much is left after running the business."],
  net_margin: ["Net margin", "Of each $1 of sales, how much ends up as profit."],
  revenue_growth_yoy: ["Sales growth", "How much sales grew (or shrank) compared with the year before."],
  roe: ["Return on equity", "Profit made for every $1 that shareholders own of the business."],
  current_ratio: ["Current ratio", "Cash and short-term assets for every $1 of bills due within a year. Above 1 means it can cover them."],
};

let companies = [];
let ticker = ""; // "" = all companies
const finCache = new Map();

/* ---------- helpers ---------- */

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body, timeoutMs = 90000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let resp;
  try {
    resp = await fetch(API + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (err) {
    throw new ApiError(
      err.name === "AbortError" ? "That took too long. Try again." : "Can't reach FilingLens right now. Try again in a moment.",
      0,
    );
  } finally {
    clearTimeout(timer);
  }
  if (resp.ok) return resp.json();
  let detail = "";
  try {
    const data = await resp.json();
    detail = typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : "";
  } catch {
    /* not JSON */
  }
  if (resp.status === 429) {
    const wait = resp.headers.get("Retry-After");
    throw new ApiError(`Too many questions at once. Try again${wait ? ` in ${wait} seconds` : " in a minute"}.`, 429);
  }
  if (resp.status === 422) throw new ApiError(detail || "That question couldn't be read. Check it and try again.", 422);
  throw new ApiError(detail || `Something went wrong on our side (error ${resp.status}). Try again.`, resp.status);
}

function errorBox(message) {
  return `<div class="err" role="alert">${esc(message)}</div>`;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (label) button.textContent = busy ? label : button.dataset.label || button.textContent;
}

/** Index of `quote` in `text`: exact, then case- and whitespace-insensitive. Returns [start, end] or null. */
function findQuote(text, quote) {
  if (!quote) return null;
  const exact = text.indexOf(quote);
  if (exact >= 0) return [exact, exact + quote.length];
  const words = quote.trim().split(/\s+/).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!words.length) return null;
  const m = new RegExp(words.join("\\s+"), "i").exec(text);
  return m ? [m.index, m.index + m[0].length] : null;
}

function highlighted(text, quote) {
  const hit = findQuote(text, quote);
  if (!hit) return esc(text);
  return esc(text.slice(0, hit[0])) + "<mark>" + esc(text.slice(hit[0], hit[1])) + "</mark>" + esc(text.slice(hit[1]));
}

function paragraphs(html) {
  return html
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => `<p>${p.replace(/\n/g, "<br/>")}</p>`)
    .join("");
}

function itemLabel(item) {
  return esc(ITEM_NAMES[item] || `Item ${item}`);
}

function edgarUrl(cik, accession) {
  const c = String(Number(cik));
  return `https://www.sec.gov/Archives/edgar/data/${c}/${accession.replace(/-/g, "")}/${accession}-index.htm`;
}

/** "JPMORGAN CHASE & CO" -> "JPMorgan Chase", "Tesla, Inc." -> "Tesla" (display only). */
const NAME_OVERRIDES = { JPM: "JPMorgan Chase", KO: "Coca-Cola", AMZN: "Amazon", NVDA: "NVIDIA", MSFT: "Microsoft", AAPL: "Apple", TSLA: "Tesla" };
function shortName(c) {
  if (!c) return "";
  if (NAME_OVERRIDES[c.ticker]) return NAME_OVERRIDES[c.ticker];
  let n = (c.name || c.ticker).replace(/[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|plc|llc|holdings?)\.?$/i, "").replace(/\s+&\s*co$/i, "");
  if (n === n.toUpperCase()) n = n.toLowerCase().replace(/\b\w/g, (ch) => ch.toUpperCase());
  return n.trim();
}
const company = (t) => companies.find((c) => c.ticker === t);
const nameOf = (t) => shortName(company(t)) || t;

function yearsFor(t) {
  const c = company(t);
  return c ? [...c.years_10k].sort((a, b) => b - a) : [];
}

function fillSelect(sel, options, { keepFirst = false } = {}) {
  const first = keepFirst ? sel.options[0] : null;
  sel.innerHTML = "";
  if (first) sel.append(first);
  for (const [value, label] of options) sel.append(new Option(label, value));
}

/* ---------- tabs ---------- */

let currentTab = "ask";
function selectTab(name) {
  currentTab = name;
  for (const t of ["ask", "diff", "fin"]) {
    const on = t === name;
    $(`tab-${t}`).setAttribute("aria-selected", String(on));
    $(`panel-${t}`).hidden = !on;
  }
  history.replaceState(null, "", name === "ask" ? location.pathname : `#${name}`);
  syncPanels();
}
for (const t of ["ask", "diff", "fin"]) $(`tab-${t}`).addEventListener("click", () => selectTab(t));

/* ---------- company picker ---------- */

function renderPicker() {
  const pill = (t, label, logo) =>
    `<button type="button" role="radio" class="co${t ? "" : " co--all"}" data-t="${esc(t)}" aria-checked="${t === ticker}">${logo ? `<span class="co__logo" aria-hidden="true">${esc(logo)}</span>` : ""}${esc(label)}</button>`;
  $("picker").innerHTML =
    pill("", "All companies") + companies.map((c) => pill(c.ticker, shortName(c), shortName(c).charAt(0))).join("");
  $("picker").querySelectorAll(".co").forEach((b) => b.addEventListener("click", () => pickCompany(b.dataset.t)));
  // arrow keys move between pills, like a radio group
  $("picker").addEventListener("keydown", (e) => {
    if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) return;
    const pills = [...$("picker").querySelectorAll(".co")];
    const i = pills.indexOf(document.activeElement);
    if (i < 0) return;
    e.preventDefault();
    const next = pills[(i + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : pills.length - 1)) % pills.length];
    next.focus();
    pickCompany(next.dataset.t);
  });
}

function pickCompany(t) {
  if (t === ticker) return;
  ticker = t;
  $("picker").querySelectorAll(".co").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.t === t)));
  refreshAskYears();
  refreshDiffYears();
  renderExamples();
  $("introTitle").textContent = t ? `What do you want to know about ${nameOf(t)}?` : "What do you want to know?";
  $("diffResult").innerHTML = "";
  syncPanels();
}

/** Show or hide the "pick a company" prompts, and load numbers when that tab is open. */
function syncPanels() {
  const none = !ticker;
  $("diffNeed").hidden = !none;
  $("diffForm").hidden = none;
  $("finNeed").hidden = !none;
  if (none) $("finResult").innerHTML = "";
  else if (currentTab === "fin") loadFinancials();
}

function refreshAskYears() {
  const years = ticker ? yearsFor(ticker) : [...new Set(companies.flatMap((c) => c.years_10k))].sort((a, b) => b - a);
  fillSelect($("askYear"), years.map((y) => [y, `${y} report`]), { keepFirst: true });
}

function refreshDiffYears() {
  const years = yearsFor(ticker);
  fillSelect($("diffA"), years.map((y) => [y, String(y)]));
  fillSelect($("diffB"), years.map((y) => [y, String(y)]));
  if (years.length > 1) {
    $("diffB").value = String(years[0]);
    $("diffA").value = String(years[1]);
  }
}

function renderExamples() {
  let ex;
  if (ticker) {
    const n = nameOf(ticker);
    ex = [
      [ticker, `What are the biggest risks to ${n}?`, "Reads the risks section"],
      [ticker, `Is ${n} more profitable than last year?`, "Uses the reported numbers"],
      [ticker, `What new risks did ${n} add this year?`, "Compares two reports"],
      [ticker, `Is ${n} involved in any lawsuits?`, "Reads the legal section"],
    ];
  } else {
    const pick = (t) => company(t) || companies[0];
    const a = pick("AAPL");
    const b = pick("TSLA");
    const c = pick("KO");
    const d = pick("NVDA");
    ex = [
      [a.ticker, `Does ${shortName(a)} depend on China?`, shortName(a)],
      [b.ticker, `Is ${shortName(b)} getting sued?`, shortName(b)],
      [c.ticker, `How much money did ${shortName(c)} make last year?`, shortName(c)],
      [d.ticker, `What worries ${shortName(d)} the most?`, shortName(d)],
    ];
  }
  $("examples").innerHTML = ex
    .map(([t, q, note]) => `<button type="button" class="sugg" data-ticker="${esc(t)}" data-q="${esc(q)}">${esc(q)}<small>${esc(note)}</small></button>`)
    .join("");
  $("examples").querySelectorAll(".sugg").forEach((b) =>
    b.addEventListener("click", () => {
      pickCompany(b.dataset.ticker);
      $("askQ").value = b.dataset.q;
      $("askForm").requestSubmit();
    }),
  );
}

async function boot() {
  try {
    const [ready, list] = await Promise.all([api("/ready", { timeoutMs: 60000 }), api("/companies", { timeoutMs: 60000 })]);
    companies = list.filter((c) => c.years_10k.length);
    if (!companies.length) throw new ApiError("No companies are loaded yet.", 0);
    $("status").textContent = `Ready · ${companies.length} companies`;
    $("status").classList.add("is-ok");
    const wanted = new URLSearchParams(location.search).get("company")?.toUpperCase();
    if (wanted && company(wanted)) ticker = wanted;
    renderPicker();
    refreshAskYears();
    refreshDiffYears();
    renderExamples();
    if (ticker) $("introTitle").textContent = `What do you want to know about ${nameOf(ticker)}?`;
    syncPanels();
    if (ready.provider === "none") {
      showBanner("Answers are built only from sentences quoted word for word from the reports, and numbers come from reported data. Nothing is written or invented by an AI.");
    }
  } catch (err) {
    $("status").textContent = "Offline";
    $("status").classList.add("is-down");
    $("picker").innerHTML = '<span class="picker__loading">Companies will appear once FilingLens is reachable.</span>';
    showBanner(err.message, true);
  }
}

function showBanner(text, isError = false) {
  const b = $("banner");
  b.textContent = text;
  b.classList.toggle("is-error", isError);
  b.hidden = false;
}

/* ---------- Ask ---------- */

let lastCitations = [];

$("askQ").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("askForm").requestSubmit();
  }
});

$("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("askQ").value.trim();
  if (question.length < 3) return;
  const btn = $("askBtn");
  btn.dataset.label = "Ask";
  setBusy(btn, true, "Reading…");
  $("askIntro").classList.add("is-compact");
  $("examples").classList.add("is-hidden");
  $("askResult").hidden = false;
  $("qBubble").textContent = question;
  $("ansMeta").innerHTML = "";
  $("ansText").innerHTML = '<p class="loading">Reading the reports and checking every quote</p>';
  $("howSteps").innerHTML = "";
  $("srcHead").textContent = "Tap a numbered tag in the answer to see the sentence it came from.";
  $("srcBody").innerHTML = "";
  $("citeList").innerHTML = "";
  const body = { question, mode: "agent" };
  if (ticker) body.ticker = ticker;
  if ($("askYear").value) body.fiscal_year = Number($("askYear").value);
  try {
    renderAnswer(await api("/ask", { method: "POST", body }));
  } catch (err) {
    $("ansText").innerHTML = errorBox(err.message);
  } finally {
    setBusy(btn, false, "Ask");
    $("examples").classList.remove("is-hidden");
  }
});

function whereLabel(c) {
  if (c.source === "tool") return "Reported numbers";
  return `${esc(nameOf(c.ticker))} ${esc(c.fiscal_year ?? "")} report · ${itemLabel(c.item)}`;
}

function renderAnswer(resp) {
  const a = resp.answer;
  lastCitations = a.citations || [];
  const verified = lastCitations.filter((c) => c.verified).length;
  const secs = resp.latency_ms < 1000 ? "under a second" : `${(resp.latency_ms / 1000).toFixed(1)} s`;
  $("ansMeta").innerHTML = [
    `<span class="tag tag--${esc(a.confidence)}">${esc(CONFIDENCE[a.confidence] || a.confidence)}</span>`,
    lastCitations.length ? `<span class="tag">${verified} of ${lastCitations.length} quotes checked</span>` : "",
    `<span class="tag tag--quiet">Answered in ${secs}${resp.cached ? " (saved answer)" : ""}</span>`,
  ].join("");

  let html = esc(a.answer).replace(/\[(\d+)\]/g, (m, n) =>
    lastCitations.some((c) => String(c.n) === n) ? `<button type="button" class="cn" data-n="${n}" aria-label="Show source ${n}">${n}</button>` : m,
  );
  html = paragraphs(html);
  if (a.insufficient_context) html = `<div class="refuse">The reports loaded here don't answer this directly.</div>` + html;
  $("ansText").innerHTML = html;
  $("ansText").querySelectorAll(".cn").forEach((b) => b.addEventListener("click", () => showCitation(Number(b.dataset.n))));

  $("citeList").innerHTML = lastCitations
    .map(
      (c) =>
        `<li><button type="button" class="cite-row" data-n="${c.n}"><span class="cn cn--static">${c.n}</span><span>${whereLabel(c)}</span><span class="${c.verified ? "ok" : "warn"}">${c.verified ? "✓ checked" : "not found"}</span></button></li>`,
    )
    .join("");
  $("citeList").querySelectorAll(".cite-row").forEach((b) => b.addEventListener("click", () => showCitation(Number(b.dataset.n))));

  const steps = resp.steps || [];
  $("how").hidden = !steps.length;
  $("howSteps").innerHTML = steps
    .map((s) => {
      const args = s.args ? Object.entries(s.args).map(([k, v]) => `${esc(k)}=${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(", ") : "";
      return `<li><strong>${esc(STEP_NAMES[s.name] || s.name)}</strong>${args ? ` <code>${args}</code>` : ""} <span class="ms">${s.latency_ms} ms</span>${s.is_error ? ' <span class="warn">error</span>' : ""}</li>`;
    })
    .join("");

  if (lastCitations.length) showCitation(lastCitations[0].n);
  else $("srcHead").textContent = "No sources for this answer.";
}

async function showCitation(n) {
  const c = lastCitations.find((x) => x.n === n);
  if (!c) return;
  document.querySelectorAll(".cn.is-on, .cite-row.is-on").forEach((el) => el.classList.remove("is-on"));
  document.querySelectorAll(`[data-n="${n}"]`).forEach((el) => el.classList.add("is-on"));
  const head = $("srcHead");
  const body = $("srcBody");
  if (c.source === "tool") {
    head.innerHTML = `<span>[${n}] From the company's reported numbers</span>`;
    body.innerHTML = `<p class="toolval"><mark>${esc(c.quote)}</mark></p><p class="hint">This number wasn't written by the AI. It comes from the financial data the company filed, and any percentage is worked out by ordinary code.</p>`;
    return;
  }
  head.innerHTML = `<span>[${n}] ${whereLabel(c)}</span>`;
  body.innerHTML = '<p class="loading">Loading the passage</p>';
  try {
    let text;
    let url = c.filing_url;
    if (c.source === "section" && c.char_start != null) {
      const s = await api(`/filings/${encodeURIComponent(c.accession_no)}/section/${encodeURIComponent(c.item)}?start=${c.char_start}&end=${c.char_end}`);
      text = { before: s.before, main: s.text, after: s.after };
      url = s.filing_url;
    } else {
      const ch = await api(`/filings/${encodeURIComponent(c.accession_no)}/chunk/${encodeURIComponent(c.chunk_id)}`);
      text = { before: "", main: ch.text, after: "" };
      url = ch.filing_url;
    }
    body.innerHTML =
      `<div class="passage">${text.before ? `<span class="ctx">…${esc(text.before.slice(-300))}</span>` : ""}${paragraphs(highlighted(text.main, c.quote))}${text.after ? `<span class="ctx">${esc(text.after.slice(0, 300))}…</span>` : ""}</div>` +
      (c.verified ? "" : '<p class="warn">This quote wasn\'t found word for word in the passage.</p>');
    if (url) head.innerHTML += ` <a href="${esc(url)}" target="_blank" rel="noopener">Read the original report ↗</a>`;
    body.querySelector("mark")?.scrollIntoView({ block: "center", behavior: "smooth" });
  } catch (err) {
    body.innerHTML = `<blockquote>${esc(c.quote)}</blockquote>` + errorBox(err.message);
  }
}

/* ---------- What changed ---------- */

$("diffForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("diffResult");
  const btn = e.submitter;
  const yearA = Number($("diffA").value);
  const yearB = Number($("diffB").value);
  if (yearA === yearB) {
    out.innerHTML = errorBox("Pick two different years to compare.");
    return;
  }
  btn.dataset.label = "Compare";
  setBusy(btn, true, "Comparing…");
  out.innerHTML = '<p class="loading">Lining up the two reports paragraph by paragraph</p>';
  try {
    const r = await api("/compare", { method: "POST", body: { ticker, item: $("diffItem").value, year_a: Math.min(yearA, yearB), year_b: Math.max(yearA, yearB) } });
    out.innerHTML = renderDiff(r);
  } catch (err) {
    out.innerHTML = errorBox(err.message);
  } finally {
    setBusy(btn, false, "Compare");
  }
});

function describeChange(share) {
  if (share == null) return "";
  if (share >= 0.95) return "Barely touched. Almost everything is copied from the year before.";
  if (share >= 0.85) return "Light edits. Mostly the same, with some lines changed.";
  if (share >= 0.7) return "A noticeable rewrite. Worth reading what's new.";
  return "A major rewrite. Much of this section is new.";
}

function renderDiff(r) {
  const c = company(r.ticker);
  const a = r.filing_a;
  const b = r.filing_b;
  const sim = r.similarity || {};
  const share = sim.cosine_tf ?? sim.simple;
  const pct = (v) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);
  const counts = r.counts || {};
  const list = (items, cls) =>
    items.length ? `<ul class="dl ${cls}">${items.map((x) => `<li>${esc(x.text)}</li>`).join("")}</ul>` : '<p class="none">None.</p>';
  const section = ITEM_NAMES[r.item] || `Item ${r.item}`;
  return `
    <div class="verdict">
      <div class="verdict__big">${pct(share)}</div>
      <div>
        <h2>of the wording in “${esc(section)}” is the same in ${b.fiscal_year} as in ${a.fiscal_year}.</h2>
        <p>${esc(describeChange(share))}${c ? ` Read the originals: <a href="${esc(edgarUrl(c.cik, a.accession_no))}" target="_blank" rel="noopener">${a.fiscal_year} report ↗</a> · <a href="${esc(edgarUrl(c.cik, b.accession_no))}" target="_blank" rel="noopener">${b.fiscal_year} report ↗</a>` : ""}</p>
      </div>
    </div>
    <div class="tally">
      <span><b>${counts.headings_added ?? 0}</b> new headings</span>
      <span><b>${counts.headings_removed ?? 0}</b> dropped headings</span>
      <span><b>${counts.paragraphs_added ?? 0}</b> new paragraphs</span>
      <span><b>${counts.paragraphs_removed ?? 0}</b> removed paragraphs</span>
      <span><b>${counts.paragraphs_modified ?? 0}</b> reworded</span>
      <span><b>${counts.paragraphs_unchanged ?? 0}</b> unchanged</span>
    </div>
    <div class="cols">
      <section><h3 class="add">New in ${b.fiscal_year}</h3><p class="sub">Headings that appear for the first time.</p>${list(r.added_risk_headings || [], "dl--add")}</section>
      <section><h3 class="del">Dropped since ${a.fiscal_year}</h3><p class="sub">Headings that no longer appear.</p>${list(r.removed_risk_headings || [], "dl--del")}</section>
    </div>
    <details class="more"><summary>Reworded paragraphs (${(r.modified_paragraphs || []).length})</summary>
      ${(r.modified_paragraphs || []).map((p) => `<div class="pair"><div><span class="lbl">${a.fiscal_year}</span><p>${esc(p.old.text)}</p></div><div><span class="lbl">${b.fiscal_year}</span><p>${esc(p.new.text)}</p></div></div>`).join("") || '<p class="none">None.</p>'}
    </details>
    <details class="more"><summary>New paragraphs (${(r.added_paragraphs || []).length})</summary>${list(r.added_paragraphs || [], "dl--add")}</details>
    <details class="more"><summary>Removed paragraphs (${(r.removed_paragraphs || []).length})</summary>${list(r.removed_paragraphs || [], "dl--del")}</details>
    <details class="more"><summary>For the technical crowd: all five similarity measures</summary>
      <dl class="sims">
        <div><dt>Cosine (word counts)</dt><dd>${pct(sim.cosine_tf)}</dd></div>
        <div><dt>Jaccard (vocabulary)</dt><dd>${pct(sim.jaccard)}</dd></div>
        <div><dt>Edit similarity</dt><dd>${pct(sim.min_edit)}</dd></div>
        <div><dt>Line diff</dt><dd>${pct(sim.simple)}</dd></div>
        <div><dt>Meaning (embeddings)</dt><dd>${pct(sim.embedding_cosine)}</dd></div>
      </dl>
      <p class="hint">100% means identical. These are the measures used in the Lazy Prices research.</p>
    </details>
    ${r.truncated ? '<p class="hint">Long lists are shortened. Open the reports for the full text.</p>' : ""}`;
}

/* ---------- The numbers ---------- */

async function loadFinancials() {
  const t = ticker;
  const out = $("finResult");
  if (finCache.has(t)) {
    out.innerHTML = renderFinancials(finCache.get(t));
    return;
  }
  out.innerHTML = `<p class="loading">Looking up ${esc(nameOf(t))}'s reported numbers</p>`;
  try {
    const f = await api(`/financials/${encodeURIComponent(t)}`);
    finCache.set(t, f);
    if (ticker === t && currentTab === "fin") out.innerHTML = renderFinancials(f);
  } catch (err) {
    if (ticker === t) out.innerHTML = errorBox(err.message);
  }
}

function fmtValue(metric, kind, raw) {
  if (raw == null || raw === "") return "—";
  const v = Number(raw);
  if (!Number.isFinite(v)) return esc(raw);
  if (metric === "current_ratio") return `${v.toFixed(2)}×`;
  if (kind === "ratio") return `${(v * 100).toFixed(1)}%`;
  const abs = Math.abs(v);
  const sign = v < 0 ? "−" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  return `${sign}$${abs.toLocaleString()}`;
}

/** Year-on-year direction for dollar rows (computed in code from reported values). */
function change(kind, cur, prev) {
  if (kind !== "usd" || cur == null || prev == null) return "";
  const c = Number(cur);
  const p = Number(prev);
  if (!Number.isFinite(c) || !Number.isFinite(p) || p <= 0) return "";
  const pctChg = (c / p - 1) * 100;
  if (Math.abs(pctChg) < 0.05) return '<span class="chg">flat</span>';
  return `<span class="chg ${pctChg > 0 ? "up" : "down"}">${pctChg > 0 ? "▲" : "▼"} ${Math.abs(pctChg).toFixed(0)}%</span>`;
}

function renderFinancials(f) {
  const years = [...f.years].sort((x, y) => x - y);
  const latest = years[years.length - 1];
  const rows = f.rows
    .map((r) => {
      const [label, what] = METRIC_HELP[r.metric] || [r.label, ""];
      const cells = years
        .map((y, i) => {
          const v = r.values[String(y)];
          const prev = i ? r.values[String(years[i - 1])] : null;
          return `<div class="yr${y === latest ? " is-latest" : ""}"><span>${y}</span><b>${fmtValue(r.metric, r.kind, v)}</b>${change(r.kind, v, prev)}</div>`;
        })
        .join("");
      return `<div class="metric"><div><h3>${esc(label)}</h3>${what ? `<p class="metric__what">${esc(what)}</p>` : ""}${r.formula ? `<span class="metric__formula">Worked out as ${esc(r.formula.replace(/_/g, " "))}</span>` : ""}</div><div class="years">${cells}</div></div>`;
    })
    .join("");
  const sources = (f.sources || []).slice(0, 12);
  return `
    <h2 class="fin__title">${esc(nameOf(f.ticker))}, year by year</h2>
    <div class="metrics">${rows}</div>
    <p class="hint">Figures come from the financial data the company filed with its yearly reports. Percentages and changes are worked out by ordinary code, never by the AI.</p>
    ${sources.length ? `<details class="more"><summary>Where these numbers come from (${(f.sources || []).length} sources)</summary><ul class="srclist">${sources.map((s) => `<li>${esc(Object.entries(s).filter(([, v]) => v != null && typeof v !== "object").map(([k, v]) => `${k}: ${v}`).join(" · "))}</li>`).join("")}</ul></details>` : ""}`;
}

/* ---------- start ---------- */

if (location.hash === "#diff" || location.hash === "#fin") selectTab(location.hash.slice(1));
boot();

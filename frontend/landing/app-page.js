/* FilingLens app page: talks to the FilingLens API. No dependencies.
 * Every string from the API is escaped before it is inserted; the only markup added is ours
 * (citation buttons and <mark> highlights). */

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);

const ITEM_NAMES = { "1": "Business", "1A": "Risk Factors", "3": "Legal Proceedings", "7": "MD&A", "7A": "Market Risk" };
const STEP_NAMES = {
  search_filings: "Searched the filings",
  compare_sections: "Compared two years",
  get_financials: "Read reported XBRL values",
  compute_ratio: "Computed a ratio from XBRL values",
  route: "Chose the tools",
  write: "Wrote the answer",
};

let companies = [];

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
      err.name === "AbortError" ? "The server took too long to answer. Try again." : "Can't reach the FilingLens API right now. Try again in a moment.",
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
    throw new ApiError(`You're asking quickly. Try again${wait ? ` in ${wait} s` : " in a minute"}.`, 429);
  }
  if (resp.status === 422) throw new ApiError(detail || "That request isn't valid. Check the question and try again.", 422);
  throw new ApiError(detail || `Something went wrong (HTTP ${resp.status}).`, resp.status);
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
  const name = ITEM_NAMES[item];
  return `Item ${esc(item)}${name ? ` · ${esc(name)}` : ""}`;
}

function edgarUrl(cik, accession) {
  const c = String(Number(cik));
  return `https://www.sec.gov/Archives/edgar/data/${c}/${accession.replace(/-/g, "")}/${accession}-index.htm`;
}

/* ---------- tabs ---------- */

function selectTab(name) {
  for (const t of ["ask", "diff", "fin"]) {
    const on = t === name;
    $(`tab-${t}`).setAttribute("aria-selected", String(on));
    $(`panel-${t}`).hidden = !on;
  }
  history.replaceState(null, "", name === "ask" ? location.pathname : `#${name}`);
}
for (const t of ["ask", "diff", "fin"]) $(`tab-${t}`).addEventListener("click", () => selectTab(t));

/* ---------- companies and selectors ---------- */

function fillSelect(sel, options, { keepFirst = false } = {}) {
  const first = keepFirst ? sel.options[0] : null;
  sel.innerHTML = "";
  if (first) sel.append(first);
  for (const [value, label] of options) sel.append(new Option(label, value));
}

function companyOptions() {
  return companies.map((c) => [c.ticker, `${c.ticker} · ${c.name || c.cik}`]);
}

function yearsFor(ticker) {
  const c = companies.find((x) => x.ticker === ticker);
  return c ? [...c.years_10k].sort((a, b) => b - a) : [];
}

function refreshAskYears() {
  const years = $("askTicker").value
    ? yearsFor($("askTicker").value)
    : [...new Set(companies.flatMap((c) => c.years_10k))].sort((a, b) => b - a);
  fillSelect($("askYear"), years.map((y) => [y, `FY${y}`]), { keepFirst: true });
}

function refreshDiffYears() {
  const years = yearsFor($("diffTicker").value);
  fillSelect($("diffA"), years.map((y) => [y, `FY${y}`]));
  fillSelect($("diffB"), years.map((y) => [y, `FY${y}`]));
  if (years.length > 1) {
    $("diffB").value = String(years[0]);
    $("diffA").value = String(years[1]);
  }
}

function renderExamples() {
  const pick = (t) => companies.find((c) => c.ticker === t) || companies[0];
  const a = pick("AAPL");
  const b = pick("JPM") || a;
  const c = pick("TSLA") || a;
  const ex = [
    [a.ticker, `What does ${a.name || a.ticker} say about supply chain risks?`],
    [b.ticker, `What legal proceedings does ${b.name || b.ticker} disclose?`],
    [c.ticker, `What was ${c.name || c.ticker}'s revenue and net margin?`],
    [a.ticker, `What new risk factors did ${a.name || a.ticker} add this year?`],
  ];
  $("examples").innerHTML =
    '<span class="examples__label">Try:</span>' +
    ex.map(([t, q]) => `<button type="button" class="chip" data-ticker="${esc(t)}" data-q="${esc(q)}">${esc(q)}</button>`).join("");
  $("examples").querySelectorAll(".chip").forEach((b) =>
    b.addEventListener("click", () => {
      $("askTicker").value = b.dataset.ticker;
      refreshAskYears();
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
    $("status").textContent = `${companies.length} companies · ${ready.chunks.toLocaleString()} passages`;
    $("status").classList.add("is-ok");
    fillSelect($("askTicker"), companyOptions(), { keepFirst: true });
    fillSelect($("diffTicker"), companyOptions());
    fillSelect($("finTicker"), companyOptions());
    refreshAskYears();
    refreshDiffYears();
    renderExamples();
    if (ready.provider === "none") {
      showBanner(
        "Answers are built from sentences quoted verbatim from the filings, and numbers come from reported XBRL data. No language model writes or invents anything.",
      );
    }
  } catch (err) {
    $("status").textContent = "Offline";
    $("status").classList.add("is-down");
    showBanner(err.message + " The app needs the FilingLens API.", true);
  }
}

function showBanner(text, isError = false) {
  const b = $("banner");
  b.textContent = text;
  b.classList.toggle("is-error", isError);
  b.hidden = false;
}

$("askTicker").addEventListener("change", refreshAskYears);
$("diffTicker").addEventListener("change", refreshDiffYears);

/* ---------- Ask ---------- */

let lastCitations = [];

$("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("askQ").value.trim();
  if (question.length < 3) return;
  const btn = $("askBtn");
  btn.dataset.label = "Ask";
  setBusy(btn, true, "Reading filings…");
  const result = $("askResult");
  result.hidden = false;
  $("ansMeta").innerHTML = "";
  $("ansText").innerHTML = '<p class="loading">Searching the filings and checking every quote…</p>';
  $("howSteps").innerHTML = "";
  $("srcHead").textContent = "Click a citation to see the passage it came from.";
  $("srcBody").innerHTML = "";
  $("citeList").innerHTML = "";
  const body = { question, mode: "agent" };
  if ($("askTicker").value) body.ticker = $("askTicker").value;
  if ($("askYear").value) body.fiscal_year = Number($("askYear").value);
  try {
    const resp = await api("/ask", { method: "POST", body });
    renderAnswer(resp);
  } catch (err) {
    $("ansText").innerHTML = errorBox(err.message);
  } finally {
    setBusy(btn, false, "Ask");
  }
});

function renderAnswer(resp) {
  const a = resp.answer;
  lastCitations = a.citations || [];
  const verified = lastCitations.filter((c) => c.verified).length;
  const meta = [
    `<span class="tag tag--${esc(a.confidence)}">${esc(a.confidence)} confidence</span>`,
    lastCitations.length ? `<span class="tag">${verified}/${lastCitations.length} quotes verified</span>` : "",
    resp.cached ? '<span class="tag">cached</span>' : "",
    `<span class="tag tag--quiet">${(resp.latency_ms / 1000).toFixed(1)} s</span>`,
  ];
  $("ansMeta").innerHTML = meta.join("");

  let html = esc(a.answer).replace(/\[(\d+)\]/g, (m, n) =>
    lastCitations.some((c) => String(c.n) === n) ? `<button type="button" class="cn" data-n="${n}" aria-label="Show source ${n}">${n}</button>` : m,
  );
  html = paragraphs(html);
  if (a.insufficient_context) {
    html = `<div class="refuse">The filings loaded here don't answer this directly.</div>` + html;
  }
  $("ansText").innerHTML = html;
  $("ansText").querySelectorAll(".cn").forEach((b) => b.addEventListener("click", () => showCitation(Number(b.dataset.n))));

  $("citeList").innerHTML = lastCitations
    .map((c) => {
      const where = c.source === "tool" ? `XBRL · ${esc(c.tool_name || "tool")}` : `${esc(c.ticker || "")} ${esc(c.form || "10-K")} FY${esc(c.fiscal_year ?? "")} · ${itemLabel(c.item)}`;
      return `<li><button type="button" class="cite-row" data-n="${c.n}"><span class="cn cn--static">${c.n}</span><span>${where}</span><span class="${c.verified ? "ok" : "warn"}">${c.verified ? "verified" : "not found"}</span></button></li>`;
    })
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
}

async function showCitation(n) {
  const c = lastCitations.find((x) => x.n === n);
  if (!c) return;
  document.querySelectorAll(".cn.is-on, .cite-row.is-on").forEach((el) => el.classList.remove("is-on"));
  document.querySelectorAll(`[data-n="${n}"]`).forEach((el) => el.classList.add("is-on"));
  const head = $("srcHead");
  const body = $("srcBody");
  if (c.source === "tool") {
    head.innerHTML = `<span>[${n}] Computed by <code>${esc(c.tool_name)}</code> from reported XBRL data</span>`;
    body.innerHTML = `<p class="toolval">Value used: <mark>${esc(c.quote)}</mark></p><p class="hint">Numbers are never generated by a model; they come from the company's XBRL financial data.</p>`;
    return;
  }
  head.innerHTML = `<span>[${n}] ${esc(c.ticker || "")} 10-K FY${esc(c.fiscal_year ?? "")} · ${itemLabel(c.item)}</span>`;
  body.innerHTML = '<p class="loading">Loading the passage…</p>';
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
    const main = highlighted(text.main, c.quote);
    body.innerHTML =
      `<div class="passage">${text.before ? `<span class="ctx">…${esc(text.before.slice(-300))}</span>` : ""}${paragraphs(main)}${text.after ? `<span class="ctx">${esc(text.after.slice(0, 300))}…</span>` : ""}</div>` +
      (c.verified ? "" : '<p class="warn">This quote wasn\'t found word for word in the passage.</p>');
    if (url) head.innerHTML += ` <a href="${esc(url)}" target="_blank" rel="noopener">Open on sec.gov ↗</a>`;
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
  const ticker = $("diffTicker").value;
  const yearA = Number($("diffA").value);
  const yearB = Number($("diffB").value);
  if (yearA === yearB) {
    out.innerHTML = errorBox("Pick two different years.");
    return;
  }
  btn.dataset.label = "Compare";
  setBusy(btn, true, "Comparing…");
  out.innerHTML = '<p class="loading">Aligning paragraphs and risk headings between the two filings…</p>';
  try {
    const r = await api("/compare", { method: "POST", body: { ticker, item: $("diffItem").value, year_a: Math.min(yearA, yearB), year_b: Math.max(yearA, yearB) } });
    out.innerHTML = renderDiff(r);
  } catch (err) {
    out.innerHTML = errorBox(err.message);
  } finally {
    setBusy(btn, false, "Compare");
  }
});

function renderDiff(r) {
  const c = companies.find((x) => x.ticker === r.ticker);
  const a = r.filing_a;
  const b = r.filing_b;
  const sim = r.similarity || {};
  const pct = (v) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);
  const list = (items, cls) =>
    items.length ? `<ul class="dl ${cls}">${items.map((x) => `<li>${esc(x.text)}</li>`).join("")}</ul>` : '<p class="none">None.</p>';
  const counts = r.counts || {};
  return `
    <div class="diffhead">
      <h2>${esc(r.ticker)} · ${itemLabel(r.item)} · FY${a.fiscal_year} → FY${b.fiscal_year}</h2>
      <p>${c ? `<a href="${esc(edgarUrl(c.cik, a.accession_no))}" target="_blank" rel="noopener">FY${a.fiscal_year} filing ↗</a> · <a href="${esc(edgarUrl(c.cik, b.accession_no))}" target="_blank" rel="noopener">FY${b.fiscal_year} filing ↗</a>` : ""}</p>
    </div>
    <dl class="sims">
      <div><dt>Cosine (word counts)</dt><dd>${pct(sim.cosine_tf)}</dd></div>
      <div><dt>Jaccard (vocabulary)</dt><dd>${pct(sim.jaccard)}</dd></div>
      <div><dt>Edit similarity</dt><dd>${pct(sim.min_edit)}</dd></div>
      <div><dt>Line diff (simple)</dt><dd>${pct(sim.simple)}</dd></div>
      <div><dt>Meaning (embeddings)</dt><dd>${pct(sim.embedding_cosine)}</dd></div>
    </dl>
    <p class="counts">${counts.headings_added ?? 0} headings added · ${counts.headings_removed ?? 0} removed · ${counts.paragraphs_added ?? 0} paragraphs added · ${counts.paragraphs_removed ?? 0} removed · ${counts.paragraphs_modified ?? 0} rewritten · ${counts.paragraphs_unchanged ?? 0} unchanged</p>
    <div class="cols">
      <section><h3 class="add">Added headings (FY${b.fiscal_year})</h3>${list(r.added_risk_headings || [], "dl--add")}</section>
      <section><h3 class="del">Removed headings (FY${a.fiscal_year})</h3>${list(r.removed_risk_headings || [], "dl--del")}</section>
    </div>
    <details class="more"><summary>Rewritten paragraphs (${(r.modified_paragraphs || []).length})</summary>
      ${(r.modified_paragraphs || []).map((p) => `<div class="pair"><div><span class="lbl">FY${a.fiscal_year}</span><p>${esc(p.old.text)}</p></div><div><span class="lbl">FY${b.fiscal_year}</span><p>${esc(p.new.text)}</p></div></div>`).join("") || '<p class="none">None.</p>'}
    </details>
    <details class="more"><summary>New paragraphs (${(r.added_paragraphs || []).length})</summary>${list(r.added_paragraphs || [], "dl--add")}</details>
    <details class="more"><summary>Removed paragraphs (${(r.removed_paragraphs || []).length})</summary>${list(r.removed_paragraphs || [], "dl--del")}</details>
    ${r.truncated ? '<p class="hint">Long lists are shortened; open the filings for the full text.</p>' : ""}`;
}

/* ---------- Financials ---------- */

$("finForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("finResult");
  const btn = e.submitter;
  btn.dataset.label = "Show financials";
  setBusy(btn, true, "Loading…");
  out.innerHTML = '<p class="loading">Reading reported XBRL values…</p>';
  try {
    out.innerHTML = renderFinancials(await api(`/financials/${encodeURIComponent($("finTicker").value)}`));
  } catch (err) {
    out.innerHTML = errorBox(err.message);
  } finally {
    setBusy(btn, false, "Show financials");
  }
});

function fmtValue(kind, raw) {
  if (raw == null || raw === "") return "—";
  const v = Number(raw);
  if (!Number.isFinite(v)) return esc(raw);
  if (kind === "ratio") return `${(v * 100).toFixed(1)}%`;
  const abs = Math.abs(v);
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
}

function renderFinancials(f) {
  const years = f.years;
  const rows = f.rows
    .map((r) => `<tr><th scope="row">${esc(r.label)}${r.formula ? `<span class="formula">${esc(r.formula)}</span>` : ""}</th>${years.map((y) => `<td>${fmtValue(r.kind, r.values[String(y)])}</td>`).join("")}</tr>`)
    .join("");
  const sources = (f.sources || []).slice(0, 12);
  return `
    <h2 class="fin__title">${esc(f.ticker)} · reported financials</h2>
    <div class="tablewrap"><table class="fin">
      <thead><tr><th scope="col">Metric</th>${years.map((y) => `<th scope="col">FY${y}</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="hint">Values come from the company's XBRL filings (as reported, latest filing for each year). Ratios are computed in code from those values, never by a model.</p>
    ${sources.length ? `<details class="more"><summary>Sources (${(f.sources || []).length})</summary><ul class="srclist">${sources.map((s) => `<li>${esc(Object.entries(s).filter(([, v]) => v != null && typeof v !== "object").map(([k, v]) => `${k}: ${v}`).join(" · "))}</li>`).join("")}</ul></details>` : ""}`;
}

/* ---------- start ---------- */

if (location.hash === "#diff" || location.hash === "#fin") selectTab(location.hash.slice(1));
boot();

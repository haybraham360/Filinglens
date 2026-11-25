/* FilingLens landing page interactions. No dependencies. */
(() => {
  "use strict";

  const root = document.documentElement;
  root.classList.add("js");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, reduced ? 0 : ms));

  /* ---------- page load + nav ---------- */
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.add("is-loaded")));
  const nav = document.querySelector(".nav");
  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- hero demo ---------- */
  // Fictional company. All filing text and figures are illustrative.
  const ACC = "0001987654-25-000012";
  const EDGAR = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K";

  const DEMOS = [
    {
      q: "What are Halvorsen's main supply chain risks?",
      tool: null,
      answer: [
        "The company states that most of its sensor modules are assembled by two contract manufacturers in Malaysia, and that a disruption at either facility would delay shipments for at least one quarter ",
        { cite: 1 },
        ". It also buys a specialised gallium-nitride wafer from a single supplier and has not qualified an alternative ",
        { cite: 2 },
        ". Rising tariffs on these imported components are named as a further cost risk ",
        { cite: 3 },
        ".",
      ],
      meta: { cites: 3, conf: "high" },
      source: {
        item: "Item 1A. Risk Factors",
        html: `
          <h5>We rely on a limited number of contract manufacturers.</h5>
          <p>We do not own manufacturing facilities. <span class="q" data-c="1"><span class="qn">1</span>Substantially all of our sensor modules are assembled by two contract manufacturers located in Penang, Malaysia, and a disruption at either facility could delay our shipments for one or more quarters.</span> Transitioning to a new manufacturer would require significant time and qualification costs.</p>
          <h5>We depend on a single supplier for gallium-nitride wafers.</h5>
          <p><span class="q" data-c="2"><span class="qn">2</span>We purchase a specialised gallium-nitride wafer from one supplier and have not qualified an alternative source.</span> Any interruption in supply could delay shipments for at least one quarter.</p>
          <h5>Changes in trade policy could increase our costs.</h5>
          <p><span class="q" data-c="3"><span class="qn">3</span>Tariffs on components imported from China, Malaysia and Vietnam may increase our cost of revenue,</span> and we may not be able to pass these costs on to customers.</p>`,
      },
    },
    {
      q: "How did operating margin change from FY2022 to FY2024?",
      tool: ["get_financials", "compute_ratio", "search_filings"],
      answer: [
        "Operating margin fell from ",
        { num: "18.4%" },
        " in FY2022 to ",
        { num: "14.9%" },
        " in FY2024, computed from reported operating income and revenue. Management attributes the decline mainly to higher wafer costs ",
        { cite: 1 },
        " and to expanding research and development headcount ahead of the HX-3 launch ",
        { cite: 2 },
        ".",
      ],
      meta: { cites: 2, conf: "high", tools: 3 },
      source: {
        item: "Item 7. Management's Discussion and Analysis",
        html: `
          <h5>Results of operations</h5>
          <p>Net revenue increased 8% compared with the prior fiscal year, driven by higher unit volumes in our industrial segment. <span class="q" data-c="1"><span class="qn">1</span>Gross margin declined primarily as a result of higher wafer costs, which increased following the renegotiation of our supply agreement.</span></p>
          <p>Research and development expense increased by $61 million. <span class="q" data-c="2"><span class="qn">2</span>The increase reflects the expansion of our research and development headcount in preparation for the HX-3 product launch.</span></p>
          <p>Selling, general and administrative expenses were broadly flat as a percentage of revenue.</p>`,
      },
    },
    {
      q: "What will Halvorsen's revenue be next year?",
      tool: null,
      refuse:
        "The filings don't contain a revenue forecast for FY2025, so FilingLens won't produce one. The FY2024 10-K reports results and describes risks, but management gives no numeric guidance.",
      meta: { cites: 0, conf: "insufficient context" },
      source: {
        item: "Item 7. Management's Discussion and Analysis",
        html: `
          <h5>Outlook</h5>
          <p>We expect demand in our industrial segment to remain subject to customer inventory cycles. We do not provide annual revenue guidance.</p>
          <p>Forward-looking statements in this report involve risks and uncertainties, including those described under Item 1A, Risk Factors.</p>`,
      },
    },
  ];

  const $ = (id) => document.getElementById(id);
  const askText = $("askText");
  const caret = document.querySelector(".caret");
  const toolRow = $("toolRow");
  const answerEl = $("answer");
  const metaEl = $("meta");
  const sourceHead = $("sourceHead");
  const sourceText = $("sourceText");
  const chips = document.querySelectorAll(".chip");

  let runId = 0;

  function setSource(d) {
    sourceHead.innerHTML = `<span>${d.source.item}</span><a href="${EDGAR}" target="_blank" rel="noopener">Open on sec.gov</a>`;
    sourceText.innerHTML = d.source.html;
    sourceText.classList.remove("is-swap");
    void sourceText.offsetWidth;
    sourceText.classList.add("is-swap");
  }

  function highlight(n, on = true) {
    const q = sourceText.querySelector(`.q[data-c="${n}"]`);
    if (!q) return;
    q.classList.toggle("is-on", on);
    if (on) {
      const top = q.offsetTop - sourceText.offsetTop - 24;
      sourceText.scrollTo({ top: Math.max(0, top), behavior: reduced ? "auto" : "smooth" });
    }
  }

  function citeButton(n) {
    const b = document.createElement("button");
    b.className = "cite w";
    b.type = "button";
    b.textContent = n;
    b.setAttribute("aria-label", `Show citation ${n} in the filing`);
    const show = () => {
      sourceText.querySelectorAll(".q").forEach((q) => q.classList.toggle("is-hot", q.dataset.c == n));
      highlight(n);
      b.classList.add("is-hot");
    };
    b.addEventListener("mouseenter", show);
    b.addEventListener("focus", show);
    b.addEventListener("click", show);
    b.addEventListener("mouseleave", () => b.classList.remove("is-hot"));
    b.addEventListener("blur", () => b.classList.remove("is-hot"));
    return b;
  }

  async function run(i) {
    const id = ++runId;
    const d = DEMOS[i];
    const alive = () => id === runId;

    chips.forEach((c, k) => c.classList.toggle("is-active", k === i));
    askText.textContent = "";
    answerEl.innerHTML = "";
    toolRow.hidden = true;
    toolRow.classList.remove("is-in");
    metaEl.hidden = true;
    metaEl.classList.remove("is-in");
    caret.classList.remove("is-off");
    setSource(d);

    // type the question
    for (const ch of d.q) {
      if (!alive()) return;
      askText.textContent += ch;
      await sleep(22 + Math.random() * 28);
    }
    caret.classList.add("is-off");
    await sleep(350);

    if (d.tool) {
      toolRow.innerHTML = `<span>Called</span>${d.tool.map((t) => `<code>${t}</code>`).join("")}`;
      toolRow.hidden = false;
      toolRow.classList.add("is-in");
      await sleep(700);
    }
    if (!alive()) return;

    if (d.refuse) {
      const p = document.createElement("span");
      p.className = "refuse w";
      p.textContent = d.refuse;
      answerEl.appendChild(p);
    } else {
      // stream the answer word by word
      let delay = 0;
      for (const part of d.answer) {
        if (!alive()) return;
        if (typeof part === "string") {
          for (const word of part.split(/(\s+)/)) {
            if (!word) continue;
            if (/^\s+$/.test(word)) { answerEl.appendChild(document.createTextNode(word)); continue; }
            const s = document.createElement("span");
            s.className = "w";
            s.textContent = word;
            s.style.animationDelay = reduced ? "0s" : `${delay}ms`;
            answerEl.appendChild(s);
            delay += 38;
          }
        } else if (part.num) {
          const s = document.createElement("span");
          s.className = "w num";
          s.textContent = part.num;
          s.style.animationDelay = reduced ? "0s" : `${delay}ms`;
          answerEl.appendChild(s);
          delay += 38;
        } else if (part.cite) {
          const b = citeButton(part.cite);
          b.style.animationDelay = reduced ? "0s" : `${delay}ms`;
          answerEl.appendChild(b);
          const n = part.cite;
          setTimeout(() => alive() && highlight(n), reduced ? 0 : delay + 150);
          delay += 160;
        }
      }
      await sleep(delay + 300);
    }
    if (!alive()) return;

    const m = d.meta;
    metaEl.innerHTML = [
      `<span>Citations <b>${m.cites}</b></span>`,
      m.cites ? `<span class="ok">All quotes verified in source</span>` : `<span>No answer without evidence</span>`,
      m.tools ? `<span>Tool calls <b>${m.tools}</b></span>` : "",
      `<span>Confidence <b>${m.conf}</b></span>`,
      `<span>Accession <b>${ACC}</b></span>`,
    ].join("");
    metaEl.hidden = false;
    metaEl.classList.add("is-in");
  }

  chips.forEach((c) => c.addEventListener("click", () => run(Number(c.dataset.q))));

  // start the demo once the hero has faded in
  setTimeout(() => run(0), reduced ? 0 : 900);

  /* ---------- what changed: view toggle + similarity meters ---------- */
  const paper = $("paper");
  const segBtns = document.querySelectorAll(".seg__btn");
  segBtns.forEach((b) =>
    b.addEventListener("click", () => {
      segBtns.forEach((x) => {
        const on = x === b;
        x.classList.toggle("is-active", on);
        x.setAttribute("aria-selected", String(on));
      });
      paper.dataset.view = b.dataset.view;
    })
  );

  function animateSims() {
    document.querySelectorAll("#sims i").forEach((bar) => (bar.style.width = `${Number(bar.dataset.v) * 100}%`));
    document.querySelectorAll("#sims b").forEach((el) => {
      const target = Number(el.dataset.v);
      if (reduced) { el.textContent = target.toFixed(2); return; }
      const start = performance.now();
      const step = (t) => {
        const p = Math.min(1, (t - start) / 1100);
        const e = 1 - Math.pow(1 - p, 3);
        el.textContent = (target * e).toFixed(2);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  /* ---------- agent trace replay ---------- */
  const steps = [...document.querySelectorAll(".tstep")];
  const runBtn = $("runTrace");
  let traceId = 0;

  async function replay() {
    const id = ++traceId;
    runBtn.disabled = true;
    steps.forEach((s) => { s.classList.remove("is-done", "is-running"); s.classList.add("is-pending"); });
    await sleep(250);
    for (const s of steps) {
      if (id !== traceId) return;
      s.classList.remove("is-pending");
      s.classList.add("is-running");
      await sleep(850);
      s.classList.remove("is-running");
      s.classList.add("is-done");
    }
    runBtn.disabled = false;
    runBtn.textContent = "Replay the agent";
  }
  runBtn.addEventListener("click", replay);
  steps.forEach((s) => s.classList.add("is-done"));

  /* ---------- trigger section animations when first visible ---------- */
  const once = new Map([
    [document.getElementById("sims"), animateSims],
    [document.querySelector(".trace"), replay],
  ]);
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const fn = once.get(e.target);
        if (fn) fn();
        io.unobserve(e.target);
      });
    },
    { threshold: 0.4 }
  );
  once.forEach((_, el) => el && io.observe(el));
})();

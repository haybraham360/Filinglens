/* FilingLens landing page: the chat demo and the "what changed" toggle. No dependencies. */
(() => {
  "use strict";

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, reduced ? 0 : ms));
  const $ = (id) => document.getElementById(id);

  const nav = document.querySelector(".nav");
  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- hero chat demo ---------- */
  // Fictional company. All filing text and figures are illustrative.
  const DEMOS = [
    {
      q: "What could go wrong for Halvorsen?",
      answer: [
        "Their biggest worry is supply. Almost all of their products are assembled by two factories in Malaysia, so a problem at either one could delay shipments for months ",
        { cite: 1 },
        ".",
      ],
      receipt: {
        where: "2024 yearly report · Risks",
        html: `We do not own manufacturing facilities. <span class="hl">Substantially all of our sensor modules are assembled by two contract manufacturers located in Penang, Malaysia, and a disruption at either facility could delay our shipments for one or more quarters.</span>`,
        ok: "Quote checked",
      },
    },
    {
      q: "Are they more profitable than two years ago?",
      answer: [
        "No. They keep ",
        { num: "14.9¢" },
        " of every $1 of sales before tax, down from ",
        { num: "18.4¢" },
        " in 2022. Management blames pricier parts ",
        { cite: 1 },
        ".",
      ],
      receipt: {
        where: "2024 yearly report · Management's explanation",
        html: `Net revenue increased 8% compared with the prior fiscal year. <span class="hl">Gross margin declined primarily as a result of higher wafer costs, which increased following the renegotiation of our supply agreement.</span>`,
        ok: "Numbers from reported data",
      },
    },
    {
      q: "What will their sales be next year?",
      answer: ["The report doesn't say, so I won't guess. Halvorsen states plainly that it gives no yearly sales forecast ", { cite: 1 }, "."],
      receipt: {
        where: "2024 yearly report · Outlook",
        html: `We expect demand in our industrial segment to remain subject to customer inventory cycles. <span class="hl">We do not provide annual revenue guidance.</span>`,
        ok: "Quote checked",
      },
    },
  ];

  const askText = $("askText");
  const caret = document.querySelector(".caret");
  const botBubble = $("botBubble");
  const typing = $("typing");
  const answerEl = $("answer");
  const receipt = $("receipt");
  const chips = document.querySelectorAll(".chip");
  let runId = 0;

  function word(text, cls, delay) {
    const s = document.createElement("span");
    s.className = cls;
    s.textContent = text;
    s.style.animationDelay = reduced ? "0s" : `${delay}ms`;
    return s;
  }

  async function run(i) {
    const id = ++runId;
    const alive = () => id === runId;
    const d = DEMOS[i];

    chips.forEach((c, k) => c.classList.toggle("is-active", k === i));
    askText.textContent = "";
    answerEl.innerHTML = "";
    botBubble.hidden = true;
    receipt.hidden = true;
    receipt.classList.remove("is-in");
    caret.classList.remove("is-off");

    for (const ch of d.q) {
      if (!alive()) return;
      askText.textContent += ch;
      await sleep(20 + Math.random() * 26);
    }
    caret.classList.add("is-off");
    await sleep(300);
    if (!alive()) return;

    botBubble.hidden = false;
    typing.hidden = false;
    await sleep(900);
    if (!alive()) return;
    typing.hidden = true;

    let delay = 0;
    for (const part of d.answer) {
      if (typeof part === "string") {
        for (const w of part.split(/(\s+)/)) {
          if (!w) continue;
          if (/^\s+$/.test(w)) { answerEl.appendChild(document.createTextNode(w)); continue; }
          answerEl.appendChild(word(w, "w", delay));
          delay += 34;
        }
      } else if (part.num) {
        answerEl.appendChild(word(part.num, "w num", delay));
        delay += 34;
      } else if (part.cite) {
        const b = word(String(part.cite), "w cite", delay);
        b.setAttribute("aria-hidden", "true");
        answerEl.appendChild(b);
        delay += 120;
      }
    }
    await sleep(delay + 250);
    if (!alive()) return;

    $("receiptWhere").textContent = d.receipt.where;
    $("receiptOk").textContent = d.receipt.ok;
    $("receiptText").innerHTML = d.receipt.html;
    receipt.hidden = false;
    receipt.classList.add("is-in");
    await sleep(350);
    if (!alive()) return;
    receipt.querySelector(".hl")?.classList.add("is-on");
    answerEl.querySelector(".cite")?.classList.add("is-hot");
  }

  chips.forEach((c) => c.addEventListener("click", () => run(Number(c.dataset.q))));
  setTimeout(() => run(0), reduced ? 0 : 500);

  /* ---------- what changed: view toggle ---------- */
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
    }),
  );
})();

"""FilingLens Streamlit app (specs/06): ask, see what changed, and check the numbers."""

from __future__ import annotations

import html
import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("FILINGLENS_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=5.0)

st.set_page_config(page_title="FilingLens", page_icon=":mag:", layout="wide")


# --- API helpers ------------------------------------------------------------------------------


def api_get(path: str, **params: Any) -> Any:
    """GET from the API; raises httpx.HTTPError."""
    resp = httpx.get(f"{API_URL}{path}", params=params or None, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_error(exc: httpx.HTTPError) -> str:
    """Human-readable error from an API failure."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except ValueError:
            detail = exc.response.text[:200]
        if exc.response.status_code == 429:
            return f"Too many requests. {detail} Wait a minute and try again."
        return f"{detail} (HTTP {exc.response.status_code})"
    return f"Can't reach the API at {API_URL}: {exc}"


@st.cache_data(ttl=300, show_spinner=False)
def companies() -> list[dict[str, Any]]:
    """Ingested companies (cached for 5 minutes)."""
    try:
        data: list[dict[str, Any]] = api_get("/companies")
        return data
    except httpx.HTTPError:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def chunk_source(accession: str, chunk_id: str) -> dict[str, Any] | None:
    """Text behind a chunk citation."""
    try:
        data: dict[str, Any] = api_get(f"/filings/{accession}/chunk/{chunk_id}")
        return data
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def section_source(accession: str, item: str, start: int, end: int) -> dict[str, Any] | None:
    """Text behind a section-diff citation, with context."""
    try:
        data: dict[str, Any] = api_get(f"/filings/{accession}/section/{item}", start=start, end=end)
        return data
    except httpx.HTTPError:
        return None


def ask_streaming(payload: dict[str, Any], status: Any) -> dict[str, Any]:
    """POST /ask/stream, showing each tool call as it happens; returns the answer payload."""
    answer: dict[str, Any] | None = None
    event = None
    with httpx.stream("POST", f"{API_URL}/ask/stream", json=payload, timeout=TIMEOUT) as resp:
        if resp.status_code != 200:
            resp.read()
            resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                data = json.loads(line[6:])
                if event == "step" and data.get("kind") == "tool":
                    flag = " ⚠️" if data.get("is_error") else ""
                    status.write(f"Called `{data['name']}` ({data['latency_ms']} ms){flag}")
                elif event == "step":
                    status.write(f"Writing the answer with `{data['name']}`")
                elif event == "answer":
                    answer = data
                elif event == "error":
                    raise RuntimeError(data.get("detail", "Request failed"))
    if answer is None:
        raise RuntimeError("The stream ended without an answer.")
    return answer


# --- rendering ------------------------------------------------------------------------------


def highlight(text: str, quote: str) -> str:
    """HTML with the quote marked (exact match), or the escaped text if not found."""
    i = text.find(quote)
    if i < 0:
        return html.escape(text)
    return (
        html.escape(text[:i])
        + "<mark>"
        + html.escape(quote)
        + "</mark>"
        + html.escape(text[i + len(quote) :])
    )


def render_citation(c: dict[str, Any]) -> None:
    """One citation: source passage with the quote highlighted, plus the sec.gov link."""
    mark = "✓ verified" if c.get("verified") else "⚠️ quote not found in source"
    if c.get("source") == "tool":
        title = f"[{c['n']}] computed by `{c.get('tool_name')}` from XBRL · {mark}"
    else:
        title = (
            f"[{c['n']}] {c.get('ticker')} {c.get('form')} FY{c.get('fiscal_year')} "
            f"Item {c.get('item')} · {mark}"
        )
    with st.expander(title):
        if c.get("source") == "tool":
            st.markdown(f"Value from the tool result: `{c['quote']}`")
        elif c.get("chunk_id"):
            src = chunk_source(c["accession_no"], c["chunk_id"])
            if src:
                st.markdown(
                    f"<div class='src'>{highlight(src['text'], c['quote'])}</div>",
                    unsafe_allow_html=True,
                )
        elif c.get("char_start") is not None and c.get("char_end") is not None:
            src = section_source(c["accession_no"], c["item"], c["char_start"], c["char_end"])
            if src:
                body = (
                    html.escape(src["before"])
                    + "<mark>"
                    + html.escape(src["text"])
                    + "</mark>"
                    + html.escape(src["after"])
                )
                st.markdown(f"<div class='src'>…{body}…</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"> {c['quote']}")
        if c.get("filing_url"):
            st.markdown(f"[Open the filing on sec.gov ↗]({c['filing_url']})")


def render_answer(resp: dict[str, Any]) -> None:
    """Answer text, badges, citations and the reasoning panel."""
    a = resp["answer"]
    if resp.get("degraded"):
        st.warning("Today's demo budget is used up, so this answer comes from the free mode.")
    st.markdown(a["answer"])
    validity = a.get("citation_validity")
    badges = [
        f"confidence **{a['confidence']}**",
        f"citations verified **{validity:.0%}**" if validity is not None else None,
        f"via **{resp.get('model') or resp['provider']}**",
        "cached" if resp.get("cached") else None,
        f"{resp['latency_ms'] / 1000:.1f}s",
    ]
    st.caption(" · ".join(b for b in badges if b))
    if a.get("insufficient_context"):
        st.info("The filings available don't fully answer this question.")
    if resp.get("unsupported_numbers"):
        st.warning("Numbers not traced to a tool result: " + ", ".join(resp["unsupported_numbers"]))
    for c in a.get("citations", []):
        render_citation(c)
    if resp.get("steps"):
        with st.expander("How I got this answer"):
            rows = [
                {
                    "step": s["index"],
                    "kind": s["kind"],
                    "name": s["name"],
                    "ms": s["latency_ms"],
                    "args": json.dumps(s.get("args") or {}),
                    "error": s.get("is_error", False),
                }
                for s in resp["steps"]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"trace id `{resp['trace_id']}` · GET {API_URL}/traces/{resp['trace_id']}")


def fmt_value(kind: str, raw: str | None) -> str:
    """Display a metric value."""
    if raw is None:
        return "-"
    v = float(raw)
    if kind == "ratio":
        return f"{v * 100:.1f}%" if abs(v) < 20 else f"{v:.2f}"
    return f"${v / 1e9:,.2f}B" if abs(v) >= 1e9 else f"${v / 1e6:,.1f}M"


# --- page ---------------------------------------------------------------------------------

st.markdown(
    "<style>.src{font-size:0.9rem;line-height:1.5;white-space:pre-wrap;max-height:22rem;"
    "overflow:auto;padding:.6rem;border-left:3px solid #999}mark{padding:0 .15em}</style>",
    unsafe_allow_html=True,
)
st.title("FilingLens")
st.caption(
    "Answers about SEC 10-K and 10-Q filings, with every claim cited to the filing "
    "and every number computed by code from XBRL, not by a language model."
)

try:
    ready = api_get("/ready")
except httpx.HTTPError as exc:
    st.error(api_error(exc))
    st.stop()
comps = companies()
if not comps:
    st.warning(
        "No filings are loaded yet. Run `filinglens ingest`, `parse` and `embed` "
        "(or `filinglens seed-import`)."
    )

with st.sidebar:
    st.subheader("Scope")
    tickers = ["Any company"] + [c["ticker"] for c in comps]
    names = {c["ticker"]: c.get("name") or c["ticker"] for c in comps}
    ticker = st.selectbox(
        "Company", tickers, format_func=lambda t: t if t == "Any company" else f"{t} · {names[t]}"
    )
    comp = next((c for c in comps if c["ticker"] == ticker), None)
    years = sorted(set(comp["years_10k"]) | set(comp["years_10q"]), reverse=True) if comp else []
    year = st.selectbox("Fiscal year", ["Any year", *years])
    mode = st.radio(
        "Mode",
        ["agent", "rag"],
        format_func=lambda m: "Agent with tools" if m == "agent" else "Search + answer",
        help="The agent can compute ratios and diff filings between years.",
    )
    quality = (
        "high"
        if st.toggle("High quality", help="Stronger model; stricter rate limit.")
        else "standard"
    )
    st.divider()
    llm = {
        "none": "no LLM (extractive, free)",
        "ollama": "local model (Ollama)",
        "anthropic": "Claude",
    }.get(ready["provider"], ready["provider"])
    st.caption(f"Answers by: {llm} · {ready['chunks']} filing chunks indexed")

tab_ask, tab_diff, tab_fin = st.tabs(["Ask", "What changed?", "Financials"])

with tab_ask:
    example_t = comp["ticker"] if comp else (comps[0]["ticker"] if comps else "AAPL")
    ex_comp = comp or (comps[0] if comps else None)
    ex_year = max(ex_comp["years_10k"]) if ex_comp and ex_comp["years_10k"] else 2023
    examples = [
        f"What are {example_t}'s main supply chain risks?",
        f"How did {example_t}'s operating margin change from {ex_year - 2} to {ex_year}?",
        f"What new risk factors did {example_t} add in its fiscal {ex_year} 10-K?",
        f"What legal proceedings does {example_t} describe?",
    ]
    st.write("Try one:")
    cols = st.columns(len(examples))
    for col, q in zip(cols, examples, strict=True):
        if col.button(q, width="stretch"):
            st.session_state["pending"] = q

    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["response"])

    question = st.chat_input("Ask about a company's filings…") or st.session_state.pop(
        "pending", None
    )
    if question:
        with st.chat_message("user"):
            st.write(question)
        payload: dict[str, Any] = {"question": question, "mode": mode, "quality": quality}
        if comp:
            payload["ticker"] = comp["ticker"]
        if year != "Any year":
            payload["fiscal_year"] = year
        with st.chat_message("assistant"):
            try:
                with st.status("Working…", expanded=False) as status:
                    resp = ask_streaming(payload, status)
                    status.update(label="Done", state="complete")
                render_answer(resp)
                history.append({"question": question, "response": resp})
            except httpx.HTTPError as exc:
                st.error(api_error(exc))
            except RuntimeError as exc:
                st.error(str(exc))

with tab_diff:
    if not comps:
        st.info("Load filings first.")
    else:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        d_ticker = c1.selectbox(
            "Company",
            [c["ticker"] for c in comps],
            key="d_t",
            index=max(0, [c["ticker"] for c in comps].index(ticker)) if comp else 0,
        )
        d_comp = next(c for c in comps if c["ticker"] == d_ticker)
        d_years = sorted(d_comp["years_10k"])
        d_item = c2.selectbox(
            "Item",
            ["1A", "7", "3", "1"],
            format_func=lambda i: {
                "1A": "1A Risk factors",
                "7": "7 MD&A",
                "3": "3 Legal",
                "1": "1 Business",
            }[i],
        )
        if len(d_years) < 2:
            st.info(f"{d_ticker} needs two 10-Ks to compare; available: {d_years or 'none'}.")
        else:
            year_b = c4.selectbox("Newer", list(reversed(d_years)), key="d_b")
            older = [y for y in d_years if y < year_b] or d_years[:1]
            year_a = c3.selectbox("Older", list(reversed(older)), key="d_a")
            if st.button("Compare", type="primary"):
                try:
                    with st.spinner("Diffing the two filings…"):
                        diff = httpx.post(
                            f"{API_URL}/compare",
                            timeout=TIMEOUT,
                            json={
                                "ticker": d_ticker,
                                "item": d_item,
                                "year_a": year_a,
                                "year_b": year_b,
                            },
                        )
                        diff.raise_for_status()
                    out = diff.json()
                    sims = out["similarity"]
                    m = st.columns(len(sims))
                    for col, (k, v) in zip(m, sims.items(), strict=True):
                        col.metric(k.replace("_", " "), "-" if v is None else f"{v:.3f}")
                    st.caption(
                        "Similarity between the two sections (1 = identical). In the "
                        "Lazy Prices paper, lower similarity predicted lower returns."
                    )
                    left, right = st.columns(2)
                    with left:
                        st.subheader(f"Removed (FY{year_a})")
                        for h in out["removed_risk_headings"] or [{"text": "None"}]:
                            st.markdown(f"- {h['text']}")
                    with right:
                        st.subheader(f"Added (FY{year_b})")
                        for h in out["added_risk_headings"] or [{"text": "None"}]:
                            st.markdown(f"- {h['text']}")
                    counts = out["counts"]
                    st.caption(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in counts.items()))
                    for title, key in (
                        ("Added paragraphs", "added_paragraphs"),
                        ("Removed paragraphs", "removed_paragraphs"),
                    ):
                        with st.expander(f"{title} ({len(out[key])})"):
                            for p in out[key]:
                                st.markdown(f"> {p['text']}")
                    with st.expander(f"Modified paragraphs ({len(out['modified_paragraphs'])})"):
                        for pair in out["modified_paragraphs"]:
                            a, b = st.columns(2)
                            a.markdown(f"**FY{year_a}**\n\n{pair['old']['text']}")
                            b.markdown(f"**FY{year_b}**\n\n{pair['new']['text']}")
                            st.divider()
                except httpx.HTTPError as exc:
                    st.error(api_error(exc))

with tab_fin:
    if not comps:
        st.info("Load filings first.")
    else:
        f_ticker = st.selectbox("Company", [c["ticker"] for c in comps], key="f_t")
        try:
            fin = api_get(f"/financials/{f_ticker}")
            yrs = [str(y) for y in fin["years"]]
            table = pd.DataFrame(
                {
                    r["label"]: [fmt_value(r["kind"], r["values"].get(y)) for y in yrs]
                    for r in fin["rows"]
                },
                index=[f"FY{y}" for y in yrs],
            ).T
            st.dataframe(table, width="stretch")
            margins = {
                r["label"]: [
                    float(r["values"][y]) * 100 if r["values"].get(y) else None for y in yrs
                ]
                for r in fin["rows"]
                if r["metric"].endswith("margin")
            }
            if margins:
                st.subheader("Margins (%)")
                st.line_chart(pd.DataFrame(margins, index=yrs))
            with st.expander("Formulas and sources"):
                for r in fin["rows"]:
                    if r.get("formula"):
                        st.markdown(f"- **{r['label']}**: `{r['formula']}`")
                st.caption(
                    "Values come from XBRL companyfacts as originally filed; each "
                    "carries its source accession number."
                )
                st.dataframe(pd.DataFrame(fin["sources"]), hide_index=True)
        except httpx.HTTPError as exc:
            st.error(api_error(exc))

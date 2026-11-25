"""Streamlit app (v1). Placeholder until the API's /ask endpoint exists (specs/06)."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("FILINGLENS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FilingLens", page_icon=":mag:")
st.title("FilingLens")
st.write("Ask questions about SEC 10-K and 10-Q filings, with citations.")

try:
    health = httpx.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"API connected (v{health['version']})")
except httpx.HTTPError:
    st.warning(f"API not reachable at {API_URL}. Start it with `make api`.")

st.text_input(
    "Question", placeholder="What are Apple's main supply chain risks in FY2023?", disabled=True
)
st.caption("Q&A arrives in the Week 3 milestone.")

# Streamlit frontend: only streamlit + httpx (pandas comes with streamlit), not the API stack.
FROM python:3.12-slim
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 "streamlit>=1.38" "httpx>=0.27" \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app frontend/app.py ./frontend/app.py
USER app
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status != 200)"
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

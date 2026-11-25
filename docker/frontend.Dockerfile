# Streamlit frontend: only needs streamlit + httpx, not the full FilingLens package.
FROM python:3.12-slim
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 "streamlit>=1.38" "httpx>=0.27" && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app frontend/app.py ./frontend/app.py
USER app
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

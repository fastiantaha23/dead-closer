FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY deadcloser ./deadcloser
COPY bench ./bench

# Hugging Face Spaces runs containers as uid 1000 and expects port 7860; Render injects its own $PORT.
RUN useradd -m -u 1000 app && mkdir -p /data && chown app /data
USER app
# With DATABASE_URL set (Neon/Supabase) all memory is in Postgres; the SQLite paths are only the local fallback.
ENV DEADCLOSER_DB=/data/deadcloser.db DEADCLOSER_CHECKPOINTS=/data/checkpoints.db PORT=7860

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health')"

# One worker on purpose: in-process per-user locks. Scale by running more Spaces per tenant, or move locks to Postgres advisory locks.
CMD ["sh", "-c", "uvicorn deadcloser.api:app_factory --factory --host 0.0.0.0 --port ${PORT:-7860}"]

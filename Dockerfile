FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY deadcloser ./deadcloser
COPY bench ./bench

# SQLite state lives on /data (mount a volume in production)
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
ENV DEADCLOSER_DB=/data/deadcloser.db DEADCLOSER_CHECKPOINTS=/data/checkpoints.db PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health')"

# One worker on purpose: SQLite + in-process per-lead locks. Scale by tenant sharding, or swap in Postgres.
CMD ["sh", "-c", "uvicorn deadcloser.api:app_factory --factory --host 0.0.0.0 --port ${PORT}"]

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_PATH=/opt/nevis-model RERANKER_PATH=/opt/nevis-rerankers HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nevis/__init__.py nevis/chunks.py nevis/embeddings.py nevis/reranker.py ./nevis/
RUN HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python -m nevis.embeddings \
    && HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python -m nevis.reranker \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /data && chown app:app /data
COPY nevis ./nevis
USER app
ENV DATABASE_PATH=/data/nevis.sqlite3
EXPOSE 8000
CMD ["uvicorn", "nevis.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test
USER root
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt
COPY tests ./tests
COPY pytest.ini .
COPY scripts ./scripts
USER app

FROM base AS runtime

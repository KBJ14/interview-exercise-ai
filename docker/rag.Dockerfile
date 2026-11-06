# Python RAG service (rag/api.py). The same image also runs the ingest CLI (scripts/ingest.py),
# so it carries the sample documents for seeding.
#
# Configuration comes only from the runtime environment (POSTGRES_*, OPENAI_API_KEY, RAG_*):
# no secret is passed as a build argument or written into a layer, and .env is excluded from the
# build context by .dockerignore. Inside the image there is no .env for rag/db.py to fall back on.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

# Dependencies first, so source changes do not reinstall packages.
COPY requirements.txt .
RUN pip install -r requirements.txt

# The chunker needs the cl100k_base encoding. Fetch it at build time so ingestion never downloads
# at runtime, and make the cache readable by the unprivileged user below.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX "$TIKTOKEN_CACHE_DIR"

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin rag

# Only what the service and the ingest CLI need; tests, docs and Go code stay out.
COPY rag/ rag/
COPY scripts/ingest.py scripts/
COPY data/documents/ data/documents/

USER rag
EXPOSE 8000

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

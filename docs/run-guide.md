# Run Guide (Part 1 RAG pipeline)

How to run the Python RAG pipeline from a fresh clone: database, ingestion, retrieval, and answer generation.
Commands run from the repository root unless noted.

## Prerequisites

| Requirement | Checked with | Notes |
|-------------|--------------|-------|
| Docker with Compose v2 | `docker compose version` | On WSL, enable Docker Desktop's WSL integration for your distro |
| Python 3.12 with `venv` | `python3 --version` | Tested on 3.12.3 |
| OpenAI API key | `[ -n "$OPENAI_API_KEY" ] && echo set` | Only needed for steps 7–9; never paste it on the command line or into files in the repo |

To load the key without echoing it or writing it to shell history:

```bash
read -rs -p "OpenAI API key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
```

## 1. Configure the database

```bash
cp .env.example .env
sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" .env
```

`.env` is git-ignored. Change `POSTGRES_PORT` if 5432 is already in use on your machine.

## 2. Start PostgreSQL with pgvector

```bash
docker compose up -d --wait
```

`--wait` returns once the healthcheck passes. `docker compose ps` should show the `db` service as `healthy`.

## 3. Apply the schema

The schema is not applied automatically; run it once per new database volume.

```bash
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < schema.sql
```

Expected: `CREATE TABLE` twice, `CREATE INDEX`, `COMMIT`. Running it again on the same database fails because the tables already exist.

## 4. Create the Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 5. Run the tests (no API calls)

```bash
.venv/bin/python -m unittest discover -s tests
```

All tests mock the OpenAI API. The database tests need steps 2–3; they write inside transactions that are rolled back (or delete only their own rows), and are skipped if PostgreSQL is not reachable.

## 6. Preview the chunks (no API calls)

```bash
.venv/bin/python scripts/preview_chunks.py
```

Expected: 3 documents, 5 chunks each, every chunk at most 300 tokens.
Add `--show "Refund Policy"` to print a chunk's full text.

## 7. Ingest the documents (3 embedding requests)

```bash
.venv/bin/python scripts/ingest.py data/documents/*.txt
```

Expected: `inserted ... chunks=5 api_calls=1` per file. Only `data/documents/*.txt` is corpus input; files in `docs/` are evaluation material and must not be ingested.

Running the same command again prints `skipped ... api_calls=0` for each file and makes no API calls.

## 8. Search (1 embedding request, optional)

```bash
.venv/bin/python scripts/search.py "Can I get a refund for a monthly renewal charge?"
```

Prints the top 3 chunks by cosine distance (smaller is closer; it is a distance, not a probability).

## 9. Ask a question (1 embedding + 1 generation request)

```bash
.venv/bin/python scripts/ask.py "Can I get a refund for a monthly renewal charge?"
```

Expected: `answerable: True`, an answer citing `[S1]` (`billing.txt` / Refund Policy), and token usage.
The code validates the output format and citation labels, not factual accuracy.

Optional settings: `RAG_GENERATION_MODEL` (default `gpt-6-luna`), `RAG_REASONING_EFFORT` (default `low`).

## Stopping and resetting

```bash
docker compose down      # stop; data stays in the pgdata volume
docker compose down -v   # stop and delete all stored documents and embeddings
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `docker` not found inside WSL | Docker Desktop WSL integration is off for this distro |
| `port is already allocated` on `up` | Another service uses `POSTGRES_PORT`; pick another port in `.env` |
| `relation "documents" does not exist` | Step 3 was skipped for this database volume |
| `OPENAI_API_KEY is not set` | The key is not exported in the current shell |
| `HTTP 404` / `model_not_found` from `ask.py` | The project has no access to the configured generation model; the code does not switch models automatically |

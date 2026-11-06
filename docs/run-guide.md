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

Docker Compose currently runs three services: `db`, a one-shot `migrate`, and the internal Python RAG service `rag`. The Go API and sample-document seeding are not in Compose yet; run the Go API as in step 10 and ingest as in step 7.

Build the `rag` image the first time (downloads the Python base image and packages):

```bash
docker compose build rag
```

Start and check:

```bash
docker compose up -d
docker compose wait migrate   # exit code 0 = schema ready; anything else = see step 3
docker compose ps -a          # db: healthy, migrate: Exited (0), rag: healthy (may show "starting" for a few seconds)
```

- Order: `db` becomes healthy once PostgreSQL accepts TCP connections; `migrate` then applies the migrations and exits; `rag` starts only after `migrate` exited with 0.
- Do not use `docker compose up -d --wait`: it treats the finished one-shot `migrate` container as a failure and exits with 1 even when everything succeeded. `docker compose wait migrate` returns `migrate`'s own exit code instead.
- Inside Compose the database is `db:5432`; `POSTGRES_PORT` in `.env` is only the port published on the host for local tools (tests, MCP server, `go run`).

After changing Python code, `requirements.txt` or the sample documents, rebuild and restart only `rag`:

```bash
docker compose build rag && docker compose up -d rag
```

### The `rag` container

- **Internal only.** It listens on port 8000 inside the Compose network and is not published on the host; other containers reach it at `http://rag:8000`. Its healthcheck calls `GET /health`, which touches neither the database nor OpenAI.
- **Local runs keep working.** The `.venv` tests, `uvicorn` on the host (step 10), the ingest CLI and the MCP server (step 11) use the host-published `db` port as before. If you also run `uvicorn` on the host, it is a separate process on host port 8000 and does not conflict with the container.
- **Configuration** comes from the environment at start-up: `POSTGRES_*` from `.env` (with `POSTGRES_HOST=db`, `POSTGRES_PORT=5432` set by Compose), and `OPENAI_API_KEY` from the shell environment or, if unset there, from `.env` (see `.env.example`). Without a key the service still starts and is healthy; queries fail with `configuration_error` before any OpenAI request. The key is not passed to any other service.
- **Image contents.** `.dockerignore` keeps `.env`, `.venv`, `.git` and caches out of the build context; the image holds only `rag/`, `scripts/ingest.py`, `data/documents/` and the installed packages, runs as an unprivileged user, and has the tokenizer encoding cached at build time. No key or password is a build argument or part of a layer.
- Do not run `docker compose config` without `--quiet` while the key is exported: it prints the resolved environment, key included.

Call it from inside the Compose network (the slim image has no curl):

```bash
docker compose exec -T rag python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

## 3. Apply the schema

This happens automatically when step 2 runs; which path applies depends on the data volume.

| Volume | What runs | Result |
|--------|-----------|--------|
| **New, empty** | `db` runs `schema.sql` (mounted into `/docker-entrypoint-initdb.d/`) while initializing the data directory, then `migrate` runs `migrations/*.sql` | All tables and indexes from `schema.sql`; each migration finds its change already present and does nothing |
| **Existing** (has data) | The PostgreSQL image skips `/docker-entrypoint-initdb.d/` because the data directory is not empty, so `schema.sql` is **not** re-applied; only `migrate` runs | Migrations add what an older schema lacks (e.g. `documents_created_at_id_idx`); existing tables and rows are untouched |

`migrate` (`docker/migrate.sh`) applies the files in `migrations/` in file-name order with `psql -v ON_ERROR_STOP=1` and stops at the first error. There is no table recording which migrations ran, so every migration must be safe to re-run: `001` is a no-op when the index exists with the expected definition and fails when an index of that name has a different one.

Expected `migrate` log (`docker compose logs migrate`):

```
migrate: applying 001_documents_created_at_id_idx.sql
NOTICE:  documents_created_at_id_idx already exists with the expected definition; nothing to do
migrate: done (1 file(s))
```

(`NOTICE: created documents_created_at_id_idx` instead on a database from an earlier schema.)

If `docker compose wait migrate` returns non-zero:

| Check | Command |
|-------|---------|
| The failing statement and message | `docker compose logs migrate` (e.g. `ERROR: documents_created_at_id_idx exists with a different definition: …`) |
| Whether `db` is healthy and whether first-time initialization failed | `docker compose ps -a`, `docker compose logs db` (look for `running /docker-entrypoint-initdb.d/001-schema.sql` and `init process complete`) |
| Re-run after fixing the cause | `docker compose up -d --force-recreate migrate && docker compose wait migrate` |

A failed initialization of a new volume leaves a partially initialized data directory; for a volume that holds no data yet, remove it with `docker compose down -v` and start again. Never run `down -v` on a volume whose documents you want to keep: it deletes them.

Applying the files by hand still works and is what Compose does:

```bash
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < schema.sql   # empty database only
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < migrations/001_documents_created_at_id_idx.sql
```

## 4. Create the Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 5. Run the tests (no API calls)

No test needs an OpenAI key: embeddings and generation are replaced by fakes. To make sure nothing can reach OpenAI, run without the key:

```bash
env -u OPENAI_API_KEY .venv/bin/python -m unittest discover -s tests
go test ./... && go vet ./...
```

| Needs | For |
|-------|-----|
| `.venv` with `requirements.txt` (step 4) | All Python tests |
| The tiktoken encoding, downloaded on first use and cached | Chunking and ingest tests |
| Go 1.22+ and the module download (`go.mod`) | Go tests |
| PostgreSQL running (step 2) with the schema (step 3) and `./.env` (step 1) | Database tests in Python and in `internal/store` |

The database tests leave no data behind: they use rolled-back transactions, rows they delete afterwards, or (the full query flow test) a temporary schema they drop.

**Check for skipped tests.** Without a reachable database the database tests are skipped and the run still reports `OK`:

```bash
env -u OPENAI_API_KEY .venv/bin/python -m unittest discover -s tests -v 2>&1 | grep -c '\.\.\. skipped'   # 0 when everything ran
go test -v ./internal/store/ | grep -c -- '--- SKIP'                                                   # 0 when everything ran
```

What the tests cover and why: [part5-testing.md](part5-testing.md).

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

## 10. Go API (Part 2: `POST /ingest`, `POST /query`, `GET /documents`, `DELETE /documents/{id}`)

The public API is a Go server. `POST /ingest` and `POST /query` validate requests and forward them to the Python service (`POST /internal/ingest`, `POST /internal/query`); `GET /documents` and `DELETE /documents/{id}` work directly on the same PostgreSQL database. Parsing, chunking, embedding and storing a new document happen only in Python, in one transaction.
It needs Go 1.22+; the only third-party dependency is the PostgreSQL driver `github.com/jackc/pgx/v5` (v5.7.4, the newest release that supports Go 1.22).

Start the Python service (port 8000) and the Go API (port 8080) in two terminals:

```bash
.venv/bin/uvicorn rag.api:app --host 127.0.0.1 --port 8000
go run ./cmd/api
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_ADDR` | `:8080` | Go API listen address |
| `RAG_SERVICE_URL` | `http://127.0.0.1:8000` | Base URL of the Python service |
| `RAG_TIMEOUT` | `60s` | Limit for one call to the Python service (Go duration, e.g. `30s`) |
| `DB_QUERY_TIMEOUT` | `5s` | Limit for one database query |
| `DATABASE_URL` | unset | `postgres://` URL; overrides the `POSTGRES_*` settings below |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | from `.env` | Required when `DATABASE_URL` is unset |
| `POSTGRES_HOST`, `POSTGRES_PORT` | `localhost`, `5432` (or `.env`) | Database address |

Database settings follow `rag/db.py`: environment variables win over `./.env`, so running from the repository root uses the same database as the Python service. The password is never logged; the startup log shows only `host:port/dbname`.

Without API calls:

```bash
curl -s http://127.0.0.1:8080/documents        # metadata only (id, title, source, content_hash, created_at), newest first
curl -s -H 'Content-Type: application/json' -d '{"question":"  "}' http://127.0.0.1:8080/query   # rejected by Go
```

### `POST /ingest`

Two request types, converted to the same internal request:

| Body | Fields |
|------|--------|
| `application/json` | `text` (required), `title`, `source` (a label such as a file name), `format` |
| `multipart/form-data` | `file` (required, one `.txt` file, UTF-8), `title`, `format`; the source is the file's base name |

`format` is `plain` (default) or `sectioned`:

- `plain`: any text. The whole text is one section, chunked by sentences (300 tokens, 50 overlap). The title is `title`, or the source name without its extension; one of them is required.
- `sectioned`: the layout of `data/documents/*.txt` (title line, then heading + paragraph blocks). The title is read from the text, so `title` must be omitted.

The uploaded file name is metadata only: path parts are dropped (`../../etc/a.txt` -> `a.txt`) and nothing is read from or written to the server's disk.

Responses:

| Status | Body | Meaning |
|--------|------|---------|
| `201` | `{"status":"inserted","document_id":96,"chunk_count":1}` | New document stored with its chunks |
| `200` | `{"status":"skipped","document_id":3,"chunk_count":5}` | The same text (SHA-256 of the raw text, whatever its title, source or format) is already stored. `document_id` and `chunk_count` describe **that existing document**; nothing was embedded or written and the submitted metadata was ignored |
| `400` | `invalid_request` / `invalid_document` | Bad fields, invalid UTF-8, NUL characters, empty text; or text that does not fit the chosen format |
| `413` | `document_too_large` / `request_too_large` | Text over 256 KiB (UTF-8 bytes); JSON body over 2 MiB; multipart body over 320 KiB |
| `415` | `unsupported_media_type` / `unsupported_file_type` | Other `Content-Type`; a file that is not `.txt` |
| `502`/`503`/`504` | as for `/query` | Python, OpenAI or the database failed |

The Python service enforces the same 256 KiB text limit and a 2 MiB body limit on `/internal/ingest`.

```bash
# API calls: 1 embedding request per new document; none for a duplicate
curl -s -H 'Content-Type: application/json' -d '{"text":"Orders ship in 2 days.","title":"Shipping"}' http://127.0.0.1:8080/ingest
curl -s -F file=@notes.txt -F title=Notes http://127.0.0.1:8080/ingest
curl -s -F file=@data/documents/billing.txt -F format=sectioned http://127.0.0.1:8080/ingest   # already stored: 200 skipped, no API call
```

### `DELETE /documents/{id}`

Deleting a document (no API calls; **permanent**, its embeddings are removed by `ON DELETE CASCADE`):

```bash
curl -s -i -X DELETE http://127.0.0.1:8080/documents/<id>   # 204 No Content, then 404 document_not_found if repeated
```

Invalid ids (`0`, `-1`, `abc`, values above the BIGINT maximum) get `400 invalid_document_id`. To restore a deleted sample document, ingest its file again (step 7).

With API calls (1 embedding + 1 generation request):

```bash
curl -s -H 'Content-Type: application/json' -d '{"question":"Can I get a refund for a monthly renewal charge?","k":3}' http://127.0.0.1:8080/query
```

Tests: `go test ./...` and `go vet ./...`. The Python service is replaced by `httptest` fakes (no API calls). The `internal/store` tests use the real database inside rolled-back transactions and are skipped if it is not running.

## 11. MCP server (Part 4)

`rag/mcp_server.py` exposes one MCP tool, `query_knowledge_base`, over stdio. It calls the same Python pipeline as step 9, so it needs what step 9 needs: PostgreSQL running (step 2), the schema (step 3), ingested documents (step 7), the database settings in `./.env` (step 1, read by `rag/db.py`), and `OPENAI_API_KEY` in the server process's environment. Design: [part4-design.md](part4-design.md).

Run it from the repository root with the project's virtualenv:

```bash
.venv/bin/python -m rag.mcp_server
```

It prints nothing and waits for JSON-RPC on stdin; normally an MCP client starts it. stdout is reserved for the protocol, logs go to stderr.

### Connect it to Claude Code

Register it at **local scope**: the entry is stored in your own Claude Code configuration for this project, not in the repository. Run the command from the repository root and replace the two placeholders:

- `<REPO_ROOT>`: absolute path of this repository
- `<KEY_FILE>`: a file outside the repository that exports the key, e.g. created once with

  ```bash
  mkdir -p ~/.config/openai && chmod 700 ~/.config/openai
  read -rs -p "OpenAI API key: " k; echo
  (umask 077; printf "export OPENAI_API_KEY='%s'\n" "$k" > ~/.config/openai/env); unset k
  ```

```bash
cd <REPO_ROOT>
claude mcp add --scope local --transport stdio quillbox-knowledge-base -- \
  bash -c '. "<KEY_FILE>" && cd "<REPO_ROOT>" && exec .venv/bin/python -m rag.mcp_server'
```

- The server reads the key from `<KEY_FILE>` when it starts, so the Claude Code configuration holds only the file path, never the key. The database password stays in `./.env`.
- `cd "<REPO_ROOT>"` makes the working directory explicit, so `.venv/bin/python` and the `rag` package resolve regardless of how the client starts the process.
- Do not add a project-scope `.mcp.json` with these values: it would be committed and would contain your local paths.

Check and connect:

| Where | Command | What it shows |
|-------|---------|---------------|
| Shell | `claude mcp get quillbox-knowledge-base` | Scope and `✔ Connected` from a health check (starts the server briefly; does not list tools) |
| Claude Code session | `/mcp` | The server with its tool count (1). Choose **Reconnect** for it if it was added during the session or shows as failed; or restart Claude Code from the repository root |

Once connected, the session exposes the tool as `mcp__quillbox-knowledge-base__query_knowledge_base`. Each call makes 1 embedding request and at most 1 generation request.

Remove the registration with `claude mcp remove quillbox-knowledge-base -s local`.

Tests (no database, no OpenAI; `answer()` is mocked): `.venv/bin/python -m unittest tests.test_mcp_server`.

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

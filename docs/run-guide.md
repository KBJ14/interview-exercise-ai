# Run Guide

How to run the system from a fresh clone. Steps 1–2 are all the Docker stack needs; steps 4–11 run the pieces on the host for development and testing.
Commands run from the repository root unless noted. Docker design and verification: [docker.md](docker.md).

## Prerequisites

| Requirement | Checked with | Notes |
|-------------|--------------|-------|
| Docker with Compose v2 | `docker compose version` | On WSL, enable Docker Desktop's WSL integration for your distro. Tested with Compose v5.6.1 |
| Python 3.12 with `venv` | `python3 --version` | Host runs only (steps 4–11); tested on 3.12.3 |
| Go 1.22+ | `go version` | Host runs only (steps 5, 10) |
| OpenAI API key | `[ -n "$OPENAI_API_KEY" ] && echo set` | Needed to seed a new volume and to answer questions; never paste it on the command line or into files in the repo |

To load the key without echoing it or writing it to shell history:

```bash
read -rs -p "OpenAI API key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
```

## 1. Configure the database

```bash
cp .env.example .env
sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" .env
```

`.env` is git-ignored. Change `POSTGRES_PORT` or `API_PORT` if 5432 or 8080 is already in use on your machine.

The OpenAI key: either export it in the shell that runs `docker compose` (e.g. `. ~/.config/openai/env`, or the `read` command above), or put it into `OPENAI_API_KEY=` in `.env` (then `chmod 600 .env`). The shell value wins when both are set.

## 2. Start the stack with Docker Compose

Five services: `db` (PostgreSQL + pgvector), a one-shot `migrate`, the internal Python RAG service `rag`, a one-shot `seed` that ingests the sample documents, and the Go API `api`.

Start (the first run builds the images; this downloads the Python, Go and Alpine base images, packages and Go modules):

```bash
docker compose up -d
```

Check that start-up finished:

```bash
docker compose wait migrate seed   # 0 = schema ready and samples ingested (or already there); non-zero = see below
docker compose ps -a               # db, rag, api: healthy; migrate, seed: Exited (0)
curl -s http://127.0.0.1:8080/health      # {"status":"ok"}
curl -s http://127.0.0.1:8080/documents   # the 3 sample documents
curl -s -H 'Content-Type: application/json' -d '{"question":"Can I get a refund for a monthly renewal charge?"}' \
     http://127.0.0.1:8080/query           # 1 embedding + 1 generation request
```

- Order: `db` becomes healthy once PostgreSQL accepts TCP connections; `migrate` then applies the migrations and exits; `rag` and `seed` start after `migrate` exited with 0; `api` starts once `db` is healthy, `migrate` succeeded and `rag` is healthy. Nothing waits for `seed`.
- Do not use `docker compose up -d --wait`: it exits with 1 whenever a one-shot container (`migrate`, `seed`) has exited, even with status 0. `docker compose wait migrate seed` returns their own exit status instead.
- Inside Compose the database is `db:5432` and the Python service `http://rag:8000`. `POSTGRES_PORT` and `API_PORT` in `.env` are only the ports published on the host.
- **Seeding and cost.** On a new volume `seed` makes one embedding request per sample file (3). On every later start it finds the documents by content hash and reports `skipped ... api_calls=0` without calling OpenAI. `docker compose logs seed` shows the result per file. A sample file you edit is ingested as a new document; the old version stays.
- **Healthy is not the same as working end to end.** The healthchecks (`GET /health` on `rag` and `api`) only confirm that each process answers HTTP; they call neither the database, the other service nor OpenAI. Without a key the stack is still healthy, but `seed` exits 1 on a new volume (`OPENAI_API_KEY is not set` per file), `GET /documents` is empty, and `POST /query` returns 502 `rag_service_error` (the Python side reports `configuration_error`). Set the key and run `docker compose up -d` again: `rag` and `seed` are recreated with it and `seed` ingests the samples.

If `docker compose wait migrate seed` is non-zero:

| Check | Command |
|-------|---------|
| Which one failed | `docker compose ps -a` (`Exited (1)` or another non-zero status) |
| Why `seed` failed | `docker compose logs seed` (missing key, OpenAI error, database error per file) |
| Why `migrate` failed | `docker compose logs migrate`, and step 3 |
| Retry after fixing | `docker compose up -d` (exited one-shots are started again) |

Re-running and restarting: `docker compose up -d` again starts the exited `migrate` and `seed` once more (no-op and all skipped); `docker compose down` followed by `up -d` keeps the volume, so the schema is not re-applied and nothing is re-embedded. Details: [docker.md](docker.md#re-running-and-restarting).

After changing source code, dependencies or the sample documents, rebuild and restart only the affected service:

```bash
docker compose build rag seed && docker compose up -d rag seed   # rag/, scripts/ingest.py, requirements.txt, data/documents/
docker compose build api && docker compose up -d api             # cmd/, internal/, go.mod, go.sum
```

`rag` and `seed` are built from the same Dockerfile; build both so they stay on the same code.

**Port conflicts.** The host ports are `POSTGRES_PORT` (default 5432) for `db` and `API_PORT` (default 8080, bound to 127.0.0.1) for `api`; `rag` publishes nothing. If `docker compose up` reports `port is already allocated`, for example because `go run ./cmd/api` is also listening on 8080, set another value in `.env` (e.g. `API_PORT=18080`) and run `docker compose up -d` again.

### The `api` container

- Built in two stages: a static binary from the Go 1.22 toolchain (`GOTOOLCHAIN=local`, `CGO_ENABLED=0`), copied alone into an Alpine image with CA certificates; it runs as an unprivileged user.
- Gets `API_ADDR=:8080`, `RAG_SERVICE_URL=http://rag:8000` and the `POSTGRES_*` settings with `POSTGRES_HOST=db`, `POSTGRES_PORT=5432`. It never receives the OpenAI key.
- Published on `127.0.0.1:${API_PORT}` only, so it is reachable from this machine but not from the network.
- The local `go run ./cmd/api` (step 10) still works; stop the container or change `API_PORT` if both should run at the same time.

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

With Docker, the `seed` service already ran this command inside the stack (step 2); run it on the host only when working without Compose's `seed`.

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

With Docker, `docker compose up -d` (step 2) already runs both services; the commands below run them on the host instead, for development.

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
docker compose stop      # stop the containers; start again with `docker compose up -d`
docker compose down      # remove the containers and network; data stays in the pgdata volume
docker compose down -v   # also delete the volume: all stored documents and embeddings are lost
```

The services have no restart policy: after Docker or the machine restarts, run `docker compose up -d` again.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `docker` not found inside WSL | Docker Desktop is not running, or its WSL integration is off for this distro |
| `port is already allocated` on `up` | Another process uses `POSTGRES_PORT` or `API_PORT` (e.g. a local `go run`); pick another port in `.env` |
| `docker compose up -d --wait` exits 1 although everything works | It counts exited one-shot containers as failures; use `docker compose wait migrate seed` |
| `seed` Exited (1) | See `docker compose logs seed`: usually the key is missing on a new volume |
| `relation "documents" does not exist` | The schema was not applied: check `docker compose logs db migrate` (step 3) |
| `OPENAI_API_KEY is not set` | Not exported in the shell that ran `docker compose`/the script, and not in `.env` |
| `POST /query` returns 502 `rag_service_error` | The Python service failed, most often for a missing key (`docker compose logs rag`) |
| `HTTP 404` / `model_not_found` from `ask.py` | The project has no access to the configured generation model; the code does not switch models automatically |

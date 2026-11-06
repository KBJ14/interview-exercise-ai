# Docker Compose Design and Verification

How `docker compose up -d` brings up the whole stack, what each service does, and what was verified.
Commands in context: [run-guide.md, step 2](run-guide.md#2-start-the-stack-with-docker-compose).

## Services

```
docker compose up -d
  db       pgvector/pgvector:pg17          healthy = accepts TCP connections
   └─▶ migrate  one-shot, same image          applies migrations/*.sql, exits 0
        ├─▶ rag      Python FastAPI, internal    healthy = GET /health answers
        │    └─▶ api  Go REST API, 127.0.0.1    healthy = GET /health answers   (also waits for db and migrate)
        └─▶ seed     one-shot, rag image        ingests data/documents/*.txt, exits 0 or 1
```

| Service | Image | Reachable at | Ready when | Gets the OpenAI key |
|---------|-------|--------------|------------|---------------------|
| `db` | `pgvector/pgvector:pg17` | `db:5432`; host `${POSTGRES_PORT:-5432}` for local tools | `pg_isready` over TCP | no |
| `migrate` | same as `db` | – | exits 0 | no |
| `rag` | `docker/rag.Dockerfile` | `http://rag:8000`, not published on the host | `GET /health` | yes |
| `seed` | same as `rag` | – | exits 0 | yes |
| `api` | `docker/api.Dockerfile` | host `127.0.0.1:${API_PORT:-8080}` | `GET /health` | no |

- `db` is the only service with a volume (`pgdata`). Keeping the service name `db` and the volume key `pgdata` keeps an existing volume (`<project>_pgdata`) attached.
- No service sets `container_name`, so a second Compose project (`-p <name>`) gets its own containers, network and volume.
- The host-published database port stays, because the local `.venv` tests, the stdio MCP server and `go run` connect to it.
- MCP is not a container: an MCP client starts `python -m rag.mcp_server` as a local subprocess over stdio (run guide, step 11). The README lists the three services above as the required stack.

## Schema: new and existing volumes

| Volume | `db` start-up | `migrate` |
|--------|---------------|-----------|
| empty | runs `schema.sql`, mounted into `/docker-entrypoint-initdb.d/` | every migration finds its change present and does nothing |
| has data | the image skips `/docker-entrypoint-initdb.d/` ("Skipping initialization"); `schema.sql` is never re-applied | adds what an older schema lacks |

Migrations have no history table: each must be safe to re-run, and `001` fails rather than accept an index of the same name with a different definition. The `db` healthcheck uses TCP because, while initializing an empty volume, the image runs a temporary socket-only server that a socket check would report as ready too early.

## Seeding and cost

`seed` runs the existing CLI, `python scripts/ingest.py data/documents/*.txt`, with the `rag` image and settings.

| Situation | OpenAI requests | `seed` result |
|-----------|-----------------|---------------|
| New volume, key set | 1 embedding request per sample file (3) | inserted, exit 0 |
| Documents already stored (any later start) | none: the SHA-256 check runs before embedding | skipped, `api_calls=0`, exit 0 |
| New volume, no key | none | exit 1, "OPENAI_API_KEY is not set" per file |
| A sample file was edited | 1 for that file | it is stored as a **new** document; the old version stays |

No other service calls OpenAI on start-up: the `rag` and `api` healthchecks call only their own `/health`. A question costs 1 embedding and 1 generation request.

Nothing depends on `seed`, so the API stays up even when seeding fails. Health and seeding are checked separately:

```bash
docker compose wait migrate seed   # 0 only if both one-shots exited 0
docker compose logs seed           # per-file inserted / skipped / error
```

## Configuration and secrets

- `.env` (from `.env.example`) holds the database settings, `API_PORT` and optionally `OPENAI_API_KEY`. Compose takes the key from the shell environment first, then `.env`.
- The key goes only to `rag` and `seed`. Containers get `POSTGRES_HOST=db` and `POSTGRES_PORT=5432` from Compose; `POSTGRES_PORT` in `.env` is only the host port.
- `.dockerignore` keeps `.env`, key files, `.venv`, `.git` and caches out of the build context; no secret is a build argument. The images contain no `.env`, and searches of their regular files, configuration and build history found neither the key nor the database password (see below).
- Environment variables are visible to anyone who can run `docker inspect` on this machine; Docker secrets would avoid that but need the code to read a file instead of `OPENAI_API_KEY`.
- `docker compose config` without `--quiet` prints the resolved environment, key included.

## Healthy is not working

A healthy stack only means each process answers HTTP. With no or an invalid key, or with no documents, the services are still healthy:

| Check | Without key, empty volume | With key, seeded |
|-------|---------------------------|------------------|
| `docker compose ps -a` | db, rag, api healthy; migrate Exited (0); seed **Exited (1)** | all healthy; migrate and seed Exited (0) |
| `GET /documents` | `{"documents":[]}` | 3 documents |
| `POST /query` | 502 `rag_service_error` (Python: `configuration_error`, raised before any OpenAI request) | 200 with a cited answer |

## Re-running and restarting

Observed with this Compose version (v5.6.1):

| Command | `migrate` | `seed` | Data |
|---------|-----------|--------|------|
| `docker compose up -d` again, nothing changed | its exited container is started again, no-op | started again, all skipped | unchanged |
| `docker compose up -d` after the key or other settings changed | as above | recreated and run | only missing documents added |
| `docker compose down`, then `up -d` | new container, no-op | new container, all skipped | volume kept, initialization skipped |
| `docker compose down -v` | – | – | **volume deleted**: all documents are lost |

`docker compose up -d --wait` cannot be used: it exits with 1 whenever a one-shot container has exited, even with status 0. `docker compose wait migrate seed` returns the one-shot services' own result instead.

## Verification

### Separate Compose project (`-p rag-compose-check`, its own volume and ports 55434/58080)

| Step | Result |
|------|--------|
| No key, no images: `docker compose up -d` | built `rag`, `seed` and `api`; db, rag, api healthy; migrate 0; seed exited 1 with the cause per file; `GET /documents` `[]` |
| `docker compose wait migrate seed` with seed failed | exit 1 (also with the services in the other order) |
| Key from the shell, same `up -d` | rag and seed recreated; seed inserted 3 documents and 15 chunks; the API reported 720 + 631 + 689 = 2,040 embedding tokens |
| One question through `POST /query` on the Go API | 200, `answerable: true`; monthly renewals are not refundable, duplicate charges and billing errors caused by Quillbox are refunded in full; cites `[S1]`, the `billing.txt` Refund Policy chunk. Token usage of this call was not observed |
| `up -d` again; `down` then `up -d` | migrate no-op and seed all skipped (`api_calls=0`) both times; still 3 documents and 15 chunks |
| Earlier, without a key | `/health` 200; `/documents` 200 (Go → database); invalid `/query` 400; keyless `/query` 502 (Go → Python); `api` published on 127.0.0.1 only; `rag` not published |
| Images | `rag` runs as uid 10001, `api` as uid 10002; tokenizer cache works offline; no `.env` in either image; key and database password found in no regular file (counting matches and read errors separately), build history or image configuration |

All test containers, networks and volumes were removed afterwards.

### Existing project (volume `interview-exercise-ai_pgdata`)

`docker compose up -d` recreated `db` with the same volume ("Skipping initialization"), migrate was a no-op, and seed reported the three existing documents (ids 16, 3, 15) as skipped with `api_calls=0`. Document ids, content hashes, chunk text and vector fingerprints, and the table, column, index and constraint definitions were identical before and after.

## Limitations

- No `restart` policy: after Docker or the machine restarts, the stack stays stopped until `docker compose up -d` is run again.
- `seed` never removes documents; editing a sample file adds a second document.
- Migrations rely on being re-runnable instead of a history table.
- The key is passed as an environment variable (see above).
- The stack serves one machine: no TLS, no authentication on the API.

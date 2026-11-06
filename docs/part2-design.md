# Part 2 Design (REST API)

Design and verification notes for the Go REST API in front of the Python RAG service, as of commit `d8e5aa8`.
How to run it: [run-guide.md](run-guide.md) (step 10). Part 1 pipeline: [part1-design.md](part1-design.md).

## Requirements

| README Part 2 requirement | Status | Code | Tests |
|---------------------------|--------|------|-------|
| `POST /ingest` (text body or file upload) | Done | `internal/httpapi/ingest.go`, `internal/ragclient/ingest.go`, `rag/api.py` (`/internal/ingest`), `rag/ingest.py` | `internal/httpapi/ingest_test.go`, `tests/test_api_ingest.py`, `tests/test_ingest_text.py` |
| `POST /query` (answer + source chunks) | Done | `internal/httpapi/server.go`, `internal/httpapi/validate.go`, `internal/ragclient/client.go`, `rag/api.py` (`/internal/query`) | `internal/httpapi/query_test.go`, `tests/test_api.py` |
| `GET /documents` (metadata) | Done | `internal/httpapi/documents.go`, `internal/store/documents.go` | `internal/httpapi/documents_test.go`, `internal/store/documents_integration_test.go` |
| `DELETE /documents/{id}` (document and embeddings) | Done | `internal/httpapi/documents.go`, `internal/store/documents.go` | `internal/httpapi/documents_delete_test.go`, `internal/store/documents_delete_integration_test.go` |
| Go calls Python over HTTP as an internal service | Done | `internal/ragclient/` (`/internal/query`, `/internal/ingest`) | `httptest` fake Python server in `internal/httpapi/*_test.go` |
| Standard Go layout (`cmd/`, `internal/`) | Done | `cmd/api`, `internal/config`, `internal/httpapi`, `internal/ragclient`, `internal/store` | `internal/config/config_test.go` |
| Request validation and structured errors | Done | `internal/httpapi/validate.go`, `ingest.go`, `errors.go`; error handlers in `rag/api.py` | invalid-input and error-mapping tests in both test suites |
| PostgreSQL for document metadata (same DB or separate, justified) | Done: same DB, see [Shared database](#shared-database) | `schema.sql`, `internal/store/documents.go`, `rag/ingest.py` | store integration tests, `tests/test_ingest_text.py` |

## Components and request flow

```
client ──HTTP──▶ Go API (:8080)  ──HTTP──▶ Python FastAPI (:8000, internal)  ──▶ OpenAI (embeddings, responses)
                    │                              │
                    │ GET /documents               │ /internal/ingest: one transaction
                    │ DELETE /documents/{id}       │ /internal/query:  vector search
                    └──────────────▶ PostgreSQL + pgvector ◀──────────┘
                                 (documents, embeddings)
```

| Component | Responsibility |
|-----------|----------------|
| Go API (`cmd/api`) | Public endpoints, input validation and size limits, error translation, metadata listing and deletion |
| Python FastAPI (`rag/api.py`) | Internal endpoints; parsing, chunking, embedding, retrieval, answer generation, and all writes of new documents |
| PostgreSQL + pgvector | `documents` (metadata + original text) and `embeddings` (chunks + `vector(1536)`) |

- Go never embeds, chunks or inserts documents; Python never serves the public API.
- Go sends the caller's request context to Python with `RAG_TIMEOUT` (default 60s); a client disconnect cancels the Python request too.
- Every error from either service uses `{"error": {"code": "...", "message": "..."}}`. Go forwards Python's input errors (`invalid_request`, `invalid_document`, `document_too_large`, `request_too_large`) and replaces messages of dependency failures with fixed public text; driver errors, SQLSTATE codes, addresses and Python internals stay in the server logs.

## Endpoints

### `POST /ingest`

| Body | Fields |
|------|--------|
| `application/json` | `text` (required), `title`, `source`, `format` |
| `multipart/form-data` | `file` (required, one `.txt`, UTF-8), `title`, `format`; source = the file's base name |

`format`: `plain` (default; whole text as one section, title from `title` or the source name) or `sectioned` (the `data/documents` layout; the title comes from the text, so `title` must be omitted).
The upload name is metadata only: path parts are dropped and nothing is read from or written to the server's disk.

| Status | Body / code | Meaning |
|--------|-------------|---------|
| 201 | `{"status":"inserted","document_id":…,"chunk_count":…}` | Stored a new document and its chunks |
| 200 | `{"status":"skipped","document_id":…,"chunk_count":…}` | Same text (SHA-256 of the raw text) already stored; the ids and count describe that existing document; nothing embedded or written; submitted title, source and format ignored |
| 400 | `invalid_request`, `invalid_document` | Bad or unknown fields, invalid UTF-8, NUL, empty text; text not valid for the chosen format |
| 413 | `document_too_large`, `request_too_large` | Text over 256 KiB (UTF-8 bytes); JSON body over 2 MiB; multipart body over 320 KiB |
| 415 | `unsupported_media_type`, `unsupported_file_type` | Other `Content-Type`; non-`.txt` file |

### `POST /query`

Request `{"question": "...", "k": 3}`: `question` a non-empty string, `k` an integer literal 1–20 (default 3), no other fields. Go and Python apply the same rules.

```json
{"answer": "... [S1]", "answerable": true, "citations": ["S1"],
 "sources": [{"label": "S1", "document_id": 3, "title": "...", "section": "...", "chunk_index": 3,
              "chunk": "...", "cosine_distance": 0.3775, "cited": true}]}
```

`cosine_distance` is a distance (0 = same direction, smaller = closer), not a probability. `model` and token `usage` from Python are internal and not returned.

| Status | Code | Meaning |
|--------|------|---------|
| 200 | – | Answer (also when `answerable` is false) |
| 400 / 413 | `invalid_request`, `request_too_large` | Invalid input (Python is not called), body over 64 KiB |
| 502 | `rag_unavailable`, `rag_invalid_response`, `rag_service_error`, `llm_invalid_response`, `llm_request_rejected`, `embedding_invalid_response` | Python unreachable or returned something unusable; model output failed checks |
| 503 | `llm_unavailable`, `llm_rate_limited`, `database_unavailable` | Dependency outage |
| 504 | `rag_timeout`, `llm_timeout` | Python or OpenAI did not answer in time |

### `GET /documents`

`{"documents": [{"id", "title", "source", "content_hash", "created_at"}]}`, newest first (`created_at DESC, id DESC`); an empty table gives `[]`. Original text and embeddings are not selected. Errors: 503 `database_unavailable` / `database_timeout`, 500 `internal_error`.

### `DELETE /documents/{id}`

`id` must be a canonical positive integer within BIGINT. 204 with no body when a row was deleted, 404 `document_not_found` otherwise, 400 `invalid_document_id` for bad ids, then the same 503/500 as the listing.

Other methods on these paths return 405 with `Allow`; unknown paths return 404, both in the error shape above.

## Shared database

Go and Python use the same PostgreSQL database with the same settings (`DATABASE_URL`, or `POSTGRES_*` from the environment, then `.env`).

- A document and its chunks must be written together. Keeping both tables in one database lets Python do that in one local transaction; with a separate metadata database, Go would have to create a row first and Python attach embeddings later, which can leave a document without chunks (or chunks without a document) when a step fails.
- `embeddings.document_id` references `documents.id` with `ON DELETE CASCADE`; that only works within one database.
- Ownership: Python writes new documents (it needs the embeddings anyway). Go reads metadata and deletes; neither path needs embeddings or OpenAI, so Go handles them directly instead of adding an internal Python endpoint.
- Trade-off: both services depend on one schema. A column change in `documents` must be reflected in `internal/store` and `rag/ingest.py`.

## Storage rules

- **Transaction**: Python inserts the `documents` row and all `embeddings` rows in one transaction. No transaction is open while OpenAI is called.
- **Duplicates**: the SHA-256 of the raw text is looked up before chunking and embedding; a match returns 200 `skipped` without any OpenAI call. If another request stores the same text while embeddings are being created, `ON CONFLICT (content_hash) DO NOTHING` keeps the existing row and also returns `skipped`. Only that constraint is treated as a duplicate; any other violation raises, rolls back and becomes a 500.
- **Deletion**: `DELETE FROM documents WHERE id = $1`; the cascade removes the chunks in the same statement. Existence comes from the affected row count, with no lookup beforehand.

## Verification

### Automated tests

No test calls OpenAI: answers and embeddings are mocked or replaced by fixed vectors, and Go tests use an `httptest` fake of the Python service.

| Suite | Count | Coverage |
|-------|-------|----------|
| Python (`tests/`) | 69 test methods | API status codes and error mapping, ingest formats, duplicates, concurrent duplicate, other constraint violations, rollback on storage failure, size limits |
| Go (`internal/`) | 47 top-level tests | Validation, size limits, error translation, timeouts and cancellation reaching Python, unusable Python responses, routing |
| Database integration (both) | included above | Real PostgreSQL; temporary rows inside rolled-back transactions or marked and deleted afterwards; existing corpus checked unchanged |

Run with `.venv/bin/python -m unittest discover -s tests`, `go test ./...` and `go vet ./...`. Database tests are skipped when PostgreSQL is not reachable; in the final runs none were skipped.

These counts are as of the commit named at the top; current counts and the latest runs are in [part5-testing.md](part5-testing.md#latest-results).

### Live end-to-end run with OpenAI

One run against the real Go and Python servers, PostgreSQL and OpenAI, with one short temporary document and one question:

| Step | Result |
|------|--------|
| `POST /ingest` (new temporary document) | 201 `inserted`, 1 chunk stored with a 1536-dimension `text-embedding-3-small` vector |
| `GET /documents` | 200, the temporary document listed first |
| `POST /query` about a fact in that document | 200, `answerable: true`; the answer stated the document's departure time and pier and cited `[S1]`, which was the temporary document's chunk |
| `POST /ingest` (same text again) | 200 `skipped` with the same `document_id` |
| `DELETE /documents/{id}` | 204; the document and its embeddings were gone afterwards |

- The Python service's HTTP request log showed three OpenAI requests, all successful: one embeddings request for the document, one embeddings request for the question, one responses request for the answer. The duplicate ingest made no OpenAI request.
- After cleanup, the three existing documents and their 15 chunks had the same ids, content hashes, chunk contents and vector fingerprints as before the run.
- Exact token usage and cost were not observed: the public API does not return usage and the services do not log it.

## Limitations

- Uploads accept `.txt` (UTF-8) only; PDF and other formats need their own parser.
- `plain` text is chunked by sentences only; paragraph boundaries are not preserved.
- `GET /documents` returns every document; there is no pagination. The `(created_at DESC, id DESC)` index (`documents_created_at_id_idx`) matches its sort order, but a full listing still reads every row. The plan depends on the planner's cost estimates for the data: a full listing often stays a sequential scan plus sort, and a page read with `LIMIT` is more likely to use the index.
- `RAG_TIMEOUT` (default 60s) is shared by `/query` and `/ingest`; large documents on a slow connection may need a higher value.
- Metadata queries share one `DB_QUERY_TIMEOUT` (default 5s).
- The Go server needs database settings at startup, even if only `/query` is used.
- Deletion is permanent; a deleted sample document can be restored by ingesting its file again.
- Token usage is not exposed by the public API or logged.
- `GET /documents/{id}` (single document) is not implemented; it returns 405.
- Deployment is the Docker Compose stack described in [docker.md](docker.md) (PostgreSQL, schema setup, Python, sample seeding, Go); it was added after the Part 2 work recorded here.

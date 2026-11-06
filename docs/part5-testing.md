# Part 5 Testing Strategy

What the automated tests check, why, and where their limits are, as of the commit that adds `tests/test_query_flow.py`.
How to run them and what they need: [run-guide.md, step 5](run-guide.md#5-run-the-tests-no-api-calls). Design notes: [Part 1](part1-design.md), [Part 2](part2-design.md), [Part 4](part4-design.md).

This document describes the tests as they are. It does not claim that every feature was written test-first.

## Requirements

| README Part 5 requirement | Status | Files | Representative tests |
|---------------------------|--------|-------|----------------------|
| Unit tests for the Python RAG retrieval logic, with the vector store and LLM mocked | Done | `tests/test_generator.py`, `tests/test_retriever.py` (`ValidationTest`) | `test_sources_are_labelled_in_rank_order`, `test_no_retrieved_chunks_means_no_model_call`, `test_valid_answer_maps_citations_to_sources_and_usage`, `test_invalid_outputs_raise`, `test_bad_input_is_rejected_before_calling_embeddings`, `test_question_is_stripped_before_embedding` |
| Unit tests for the Go API handlers, with the Python service mocked | Done | `internal/httpapi/query_test.go`, `internal/httpapi/ingest_test.go` (fake Python service via `httptest`); `documents_test.go`, `documents_delete_test.go` (fake store, since those handlers call the database, not Python) | `TestQuerySuccessReturnsPublicResponse`, `TestInvalidInputIs400AndNeverCallsPython`, `TestPythonErrorResponsesAreTranslated`, `TestTimeoutIs504AndCancelsThePythonCall`, `TestIngestSizeLimitsAre413`, `TestDeleteMissingDocumentIs404` |
| At least one integration test running the full query flow against a real database | Done | `tests/test_query_flow.py` | `test_query_flow_from_http_request_to_cited_answer`, `test_citation_of_an_unretrieved_source_is_rejected_end_to_end` |

No requirement is missing. The test counts below are context, not the reason a requirement counts as met: each row above is met because the listed tests exercise that behaviour.

## What is tested and why

The tests concentrate on behaviour that would silently produce wrong results or leak data if it broke:

| Area | Failure the tests guard against | Examples |
|------|---------------------------------|----------|
| Chunking | Text lost or reordered, chunks over the token limit, oversized words truncated silently | `test_no_text_is_lost_or_reordered`, `test_single_word_longer_than_max_raises_instead_of_truncating` |
| Ingestion | Half-written documents, re-embedding duplicates, a concurrent duplicate turning into an error, other constraint violations hidden as "duplicate" | `test_invalid_last_chunk_rolls_back_document_and_earlier_chunks`, `test_duplicate_is_skipped_before_embedding_and_reports_the_existing_row`, `test_concurrent_duplicate_during_embedding_is_skipped_via_content_hash_conflict`, `test_other_unique_violation_is_not_treated_as_duplicate_and_rolls_back` |
| Retrieval | Wrong order or distance, results from another embedding model, `k` not respected | `test_returns_nearest_first_with_cosine_distance`, `test_rows_from_other_embedding_models_are_excluded` |
| Generation | Citations to sources that were not supplied, document text breaking out of its `<source>` tag, a model call when nothing was retrieved | `test_invalid_outputs_raise`, `test_document_text_cannot_close_or_forge_source_tags`, `test_no_retrieved_chunks_means_no_model_call` |
| HTTP boundaries | Wrong status codes, invalid input reaching downstream services, internal messages or keys leaking, timeouts and cancellations not reaching Python | `test_failures_map_to_status_by_meaning`, `TestInvalidInputIs400AndNeverCallsPython`, `TestUnusablePythonResponsesAre502`, `TestClientCancellationReachesPython` |
| Size limits | Oversized documents accepted, limits counted in characters instead of UTF-8 bytes | `test_limit_counts_utf8_bytes_not_characters`, `TestIngestSizeLimitsAre413` |
| MCP | Wrong tool schema, `score` not `1 - cosine_distance`, internals in tool errors, non-protocol output on stdout | `test_exactly_one_tool_with_input_and_output_schema`, `test_score_is_one_minus_cosine_distance`, `test_initialize_list_and_call_with_only_protocol_messages_on_stdout` |
| Database (Go) | Wrong listing order, deletes that leave chunks behind, errors misclassified as unavailable | `TestListDocumentsOrderAndFieldsOnRealDatabase`, `TestDeleteDocumentCascadesInsideTransaction`, `TestErrorClassificationOnRealConnections` |

## Kinds of tests

Counts are test methods (Python) and top-level `Test…` functions (Go, subtests not counted), taken from the test files at this commit.

| Kind | Python | Go | Needs |
|------|--------|----|-------|
| Unit: no database, no network | 66 | 40 (`internal/config`, `internal/httpapi`) | nothing beyond the installed packages |
| Real database integration | 13: `test_ingest` (2), `test_retriever.SearchQueryTest` (3), `test_ingest_text.IngestTextDatabaseTest` (6), `test_query_flow` (2) | 7 (`internal/store`) | PostgreSQL with the schema and `./.env` settings |
| HTTP and MCP connections | MCP over a real stdio subprocess (2). FastAPI runs in-process through `TestClient`, counted as unit | The `internal/httpapi` tests make real HTTP requests to an `httptest` fake of the Python service (counted as unit above) | nothing extra |
| **Total** | **81** | **47** | |

### Where the boundaries are

Only external systems are replaced; the code the tests are about runs for real.

| Replaced (mock or fake) | Runs for real |
|-------------------------|---------------|
| OpenAI embeddings (`embed_texts`) and the OpenAI Responses client | Chunking, hashing, SQL, transactions, retrieval ranking, prompt building, output parsing and citation checks |
| The Python service, from Go's side (`httptest` server) | Go validation, size limits, error translation, context timeouts and cancellation |
| The document store, in Go handler tests (`fakeLister`) | The store's SQL and error classification, in `internal/store` against PostgreSQL |
| `answer()`, in FastAPI and MCP contract tests | The HTTP and MCP layers: schemas, status codes, serialization, error shapes |

The two ingest rollback tests in `tests/test_ingest.py` were once run against a deliberately broken setup (the write transaction disabled in a separate process): both failed, so they detect the problem they are written for.

## What fake OpenAI can and cannot show

Replacing OpenAI makes the tests free, fast and deterministic, and lets them create situations the real service rarely produces (malformed JSON, citations to unknown sources, timeouts). They prove that the code handles whatever the model returns correctly.

They cannot show that the system gives good answers: fake embeddings are hand-picked axis vectors, so semantic retrieval quality is not measured, and the fake model's answer is written by the test, so grounding, wording and refusal behaviour of the real model are not measured either.

Those were checked by hand against the real OpenAI API and are recorded separately: the 7-question baseline in [rag-baseline.md](rag-baseline.md), the Part 2 end-to-end run in [part2-design.md](part2-design.md#live-end-to-end-run-with-openai), and the Claude Code tool call in [part4-design.md](part4-design.md#live-client-check). These manual checks are not automated tests and are not part of the counts above.

## Full query flow integration test

`tests/test_query_flow.py` sends one HTTP request through the real stack, replacing only OpenAI:

```
TestClient POST /internal/query
  → generator.answer() → retriever.search()
      question embedding: fake vector                        (replaced)
      pgvector SQL: ORDER BY embedding <=> query, model filter, JOIN documents   (real)
  → label_sources / build_input                               (real prompt)
  → fake Responses client: records the prompt, returns JSON citing S1   (replaced)
  → output parsing and citation checks                        (real)
  → JSON response                                             (real serialization)
```

The three test documents are stored with the real `ingest_text()` and given unit vectors at known cosine distances from the question vector (0, 1 - 1/√2, 1).

`test_query_flow_from_http_request_to_cited_answer` checks:

- HTTP 200, `answerable: true`, the expected answer, `citations: ["S1"]`
- the question was embedded once, after trimming
- sources in nearest-first order with the right `document_id`, `title`, `label` and `cited` (only S1); distances compared with a floating-point tolerance
- S1's `chunk` equals the body stored in the database
- the prompt the fake model received contains that chunk body, the `<source id="S1" document="…">` tag and the question, and the instructions do not contain the chunk

`test_citation_of_an_unretrieved_source_is_rejected_end_to_end` makes the fake model cite `S2` when only one source was retrieved; the real citation check rejects it and the API returns 502 `llm_invalid_response`.

Guarding against real calls: the tests remove `OPENAI_API_KEY` from the environment, and constructing a real OpenAI client fails the test.

What this test does not establish: it runs the real search SQL, but the query plan was not inspected, so it does not show whether PostgreSQL used the HNSW index for it. With three rows a sequential scan is likely; the asserted order and distances hold either way.

## Isolation and cleanup

| Tests | Isolation | Cleanup |
|-------|-----------|---------|
| `test_query_flow` | A new schema `test_query_flow_<random>` per test, created from the real `schema.sql`. `rag/db.py` already prefers `DATABASE_URL`, so the test sets it to the same database with `search_path=<schema>,public`; every application connection then reads and writes only that schema, with no change to application code | `DROP SCHEMA … CASCADE`, registered with `addCleanup` right after the schema is created, so it also runs when the test fails; the cleanup checks that the schema is gone and that the real tables are unchanged |
| Retrieval SQL, ingest rollback, Go store tests | A transaction that is always rolled back (Go listing tests also shadow `documents` with an empty temporary table) | Rollback |
| Ingest and Go delete tests that must commit | Rows marked with a random value in their content or title | Deleted by content hash in `tearDown`/`t.Cleanup` |

Because the query flow schema holds only the test's own documents, its ranking does not depend on the real corpus.

## Skipped tests

Database tests skip when PostgreSQL cannot be reached (Python: `unittest.skipUnless`; Go: `t.Skipf`). The run still ends with `OK` or `ok`:

- Python without a database: 13 of 81 tests are skipped and the summary reads `OK (skipped=13)`.
- Go without a database: the database-backed `internal/store` tests and subtests report `SKIP`, and `go test` still prints `ok`.

So `OK` alone does not mean the database paths were tested. Check the skipped count (run guide, step 5) before treating a run as complete.

## Latest results

| Suite | When | Result |
|-------|------|--------|
| Python, `tests/test_query_flow.py` | Latest run, real PostgreSQL, no API key | 2 run, 2 passed, 0 skipped |
| Python, full suite | Same run | 81 run, 81 passed, 0 skipped; no request to `api.openai.com` in the log (only in-process `TestClient` requests) |
| Go, `go test ./...` and `go vet ./...` | Last run during the Part 3 sort-index work, after the last Go code change (`d8e5aa8`); not re-run for this document | Passed with the database available, no skips; 47 top-level tests in the files at this commit |

Data after the latest Python run: the three existing documents (ids 3, 15, 16) and their 15 chunks had the same content hashes, chunk text and vector fingerprints as before; no `test_query_flow_*` schema and no test rows were left.

## Limitations

- Retrieval and answer quality are covered only by the manual checks above, not by automated tests.
- HNSW index use is not verified by any test.
- Go ↔ Python compatibility is tested with hand-written copies of Python's responses in the Go tests; a field change in `rag/api.py` would not fail the Go tests (a shared contract fixture would).
- Skipped database tests do not fail a run; there is no switch to make them required.
- Tests use the development database (in temporary schemas, rolled-back transactions or marked rows), not a dedicated test database; an interrupted run could leave marked rows behind.
- Applying `schema.sql` to a fresh database is exercised by `test_query_flow`, but migrations are not tested automatically.
- `cmd/api` startup and shutdown, and MCP request cancellation, have no tests.
- There is no CI configuration; tests are run by hand.

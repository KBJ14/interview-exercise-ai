> **About this repository:** a personal practice project for learning AI development tools (such as Claude Code) while building the system below. It is not a challenge submission.
> The challenge text below is kept as the project brief.
>
> - Setup and run: [docs/run-guide.md](docs/run-guide.md)
> - Part 1 design: [docs/part1-design.md](docs/part1-design.md)
> - Part 2 design and verification: [docs/part2-design.md](docs/part2-design.md)
>
> **Database schema (Part 3).** A new database is created from [`schema.sql`](schema.sql); a database created from an earlier `schema.sql` is brought up to date with [`migrations/`](migrations/) (commands: [run guide, step 3](docs/run-guide.md#3-apply-the-schema)). Every table, constraint and index is commented in `schema.sql`.
>
> - **Two tables, one-to-many.** `documents` has one row per source (title, source, full original text, `content_hash`, `created_at`); `embeddings` has one row per chunk, linked by `document_id`. Document metadata is stored once instead of on every chunk, and listing documents never reads vectors.
> - **One database for Python and Go.** Python (`rag/ingest.py`) writes a document and all of its chunk rows (text plus vector) in one transaction, with no transaction open during the OpenAI call; Go (`internal/store`) lists and deletes. A separate metadata database would split that write across services and lose the foreign key and cascade. Trade-off: both services are coupled to one schema, so a column change must be made in both ([details](docs/part2-design.md#shared-database)).
> - **Keys and constraints.** In both tables `id` is `GENERATED ALWAYS AS IDENTITY` (the database assigns it and rejects an explicitly supplied value) and the `PRIMARY KEY` (which guarantees it is unique and not null). `embeddings.document_id` is a foreign key with `ON DELETE CASCADE`: `DELETE /documents/{id}` removes the chunks in the same statement, so no orphaned vector can be retrieved. `UNIQUE (document_id, chunk_index, embedding_model)` stops two chunks at one position. `CHECK`s reject blank titles, texts and chunks, a malformed hash, negative positions and zero token counts.
> - **`content_hash`.** SHA-256 of the exact stored text, `UNIQUE`. It could be derived from `content`, but a unique index on a 64-character value is far cheaper than comparing whole texts. Ingestion looks it up before calling OpenAI, and `ON CONFLICT (content_hash)` resolves two concurrent ingests of the same text into one row. Only byte-identical text is a duplicate, whatever its title, source or format.
> - **Chunk text and vector in one row, model name on each row.** Search reads chunk text and vector from `embeddings` and joins `documents` only for the title. `embedding_model` is a plain string per row rather than a lookup table: simple under the current one-model premise, at the cost of repeating the name; switching models means re-embedding every chunk.
> - **`vector(1536)`.** The output size of `text-embedding-3-small`. The type rejects vectors of any other length, and 1536 is below pgvector's 2000-dimension limit for HNSW on `vector`.
>
> | Index | Serves |
> |-------|--------|
> | `documents_pkey` (PK, created automatically) | Delete by id, the join from `embeddings`, foreign key checks |
> | `documents_content_hash_key` (UNIQUE, automatic) | Duplicate lookup and `ON CONFLICT (content_hash)` |
> | `embeddings_document_chunk_model_key` (UNIQUE, automatic) | Its leading `document_id` serves cascade deletes and per-document chunk lookups, so the foreign key needs no extra index |
> | `embeddings_pkey` (PK, automatic) | Row identity; not used by current queries |
> | `documents_created_at_id_idx` (B-tree, `created_at DESC, id DESC`) | The `GET /documents` sort order |
> | `embeddings_embedding_hnsw_idx` (HNSW, `vector_cosine_ops`) | `ORDER BY embedding <=> query LIMIT k`; the operator class matches cosine distance `<=>` |
>
> - **Sort index.** It matches the listing order but does not reduce the rows a full listing reads. In a check on a 20,000-row throwaway database, the full listing used a sequential scan plus sort and a `LIMIT 20` query used the index; the planner's actual choice depends on the data and its cost estimates. Pagination is not implemented yet.
> - **HNSW.** Approximate search: much faster than an exact scan on large tables, but it can miss a true nearest neighbour. `hnsw.ef_search` (default 40) sets the size of the candidate list kept during the search, trading speed for recall. Search filters on `embedding_model`; with one model nothing is filtered out, but with several models filtered rows are not replaced, so a search could return fewer than `k` results.

# Take-Home Challenge: AI Knowledge Assistant

**Role:** Intermediate AI Engineer  
**Time Budget:** 4–6 hours  
**Submission:** GitHub repository with a README covering setup, design decisions, and any trade-offs made

---

## Overview

You'll build a small but production-minded **AI-powered Knowledge Assistant** — a system that lets users query a document corpus using natural language, backed by a RAG pipeline, exposed via a REST API, and accessible as an MCP server tool.

This challenge is intentionally scoped so that a strong engineer can complete the core in 4–6 hours. Optional extensions exist for candidates who want to demonstrate additional depth — they are not required and will not penalize candidates who skip them.

---

## Part 1 — RAG Pipeline (Python) `[Core]`

Build a RAG pipeline that:

1. **Ingests** a small document corpus (provided as plain text or PDF files — use anything realistic, e.g., product FAQs, support docs, a Wikipedia excerpt)
2. **Chunks and embeds** the documents using a model of your choice (e.g., `sentence-transformers`, OpenAI embeddings, or a local model via Ollama)
3. **Stores embeddings** in a vector store — use **pgvector** (PostgreSQL extension) as the backing store
4. **Retrieves** relevant chunks for a given query using semantic search
5. **Generates** a grounded answer using an LLM of your choice (local via Ollama preferred, but OpenAI is acceptable)

**What we're evaluating:**
- Quality of chunking strategy and embedding choice
- Schema design for the `documents` and `embeddings` tables in PostgreSQL
- Retrieval relevance and prompt construction
- Code structure and clarity

---

## Part 2 — REST API (Golang) `[Core]`

Wrap your RAG pipeline with a Go REST API exposing the following endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Accept a document (text body or file upload) and add it to the corpus |
| `POST` | `/query` | Accept a natural language question, return a grounded answer + source chunks |
| `GET` | `/documents` | List all ingested documents with metadata |
| `DELETE` | `/documents/:id` | Remove a document and its embeddings |

**Requirements:**
- The Go API should call the Python RAG service over HTTP (treat it as an internal service)
- Use standard Go project structure (`cmd/`, `internal/`, etc.)
- Include at least basic request validation and structured error responses
- Use **PostgreSQL** for document metadata storage (separate from the vector store, or the same DB — your call, justify it)

**What we're evaluating:**
- REST API design and HTTP semantics
- Go code quality, idiomatic patterns, and project structure
- DB schema design for document metadata
- How you handle the Go ↔ Python service boundary

---

## Part 3 — Database Design `[Core]`

Include a `schema.sql` (or migration files) covering:

- The document metadata table(s) used by the Go API
- The embeddings/vector table(s) used by the RAG pipeline
- Appropriate indexes, constraints, and foreign keys
- A brief comment in your README explaining your schema choices and any trade-offs

**What we're evaluating:**
- Schema clarity and normalization
- Appropriate use of pgvector types (`vector(n)`)
- Index strategy for both metadata queries and ANN vector search

---

## Part 4 — MCP Server `[Core]`

Expose your RAG query capability as an **MCP tool** so it can be used by AI agents or LLM clients (e.g., Claude Desktop).

Implement a minimal MCP server (Python or Go) with at least one tool:

```
Tool: query_knowledge_base
Input: { "question": string }
Output: { "answer": string, "sources": [{ "document_id", "chunk", "score" }] }
```

The MCP server should call your RAG pipeline internally (reuse the same logic — don't duplicate it).

**What we're evaluating:**
- Understanding of MCP tool schema and protocol basics
- Clean separation between transport layer (MCP) and business logic (RAG)
- Whether the tool is actually usable by an MCP-compatible client

---

## Part 5 — Tests `[Core]`

Write tests that demonstrate TDD thinking. You don't need 100% coverage — we want to see *what* you chose to test and *why*.

**Required:**
- At least **unit tests** for your RAG retrieval logic (Python) — mock the vector store and LLM
- At least **unit tests** for your Go API handlers — mock the downstream Python service
- At least one **integration test** that runs the full query flow against a real (test) database

**What we're evaluating:**
- Test structure and naming
- Use of mocks/fakes at the right boundaries
- Whether tests actually validate meaningful behaviour (not just that functions exist)

---

## Extension — LLM Fine-Tuning `[Optional]`

> **This is optional.** Only attempt this if you finish the core challenge comfortably within the time budget.

Demonstrate familiarity with fine-tuning by doing one of the following:

**Option A — Lightweight fine-tuning:**  
Fine-tune a small open-source model (e.g., a quantized LLaMA or Mistral variant) using **LoRA/QLoRA** on a small synthetic dataset relevant to your document corpus. Document what you did, why, and what you observed — working code is a bonus but the written explanation is what's evaluated.

**Option B — Fine-tuning design doc:**  
If compute constraints make Option A impractical, write a 1–2 page design doc explaining: what model you'd choose, why, what training data you'd construct, which PEFT technique you'd use, how you'd evaluate the fine-tuned model vs. the base model, and what risks you'd watch for. This will be discussed in the follow-up interview.

---

## Deliverables

Your submission should include:

- [ ] Working code for Parts 1–5 in a single GitHub repo
- [ ] `docker-compose.yml` that brings up the full stack — PostgreSQL (with pgvector), Python RAG service, and Go API
- [ ] A seed script or `make ingest` target that loads sample documents automatically on startup
- [ ] `schema.sql` or migration files with inline comments
- [ ] API documentation (OpenAPI spec, Postman collection, or markdown)
- [ ] `README.md` covering architecture, design patterns used, trade-offs, and next steps
- [ ] (Optional) Extension write-up or code

---

## Evaluation Rubric

| Area | Weight | What we look for |
|------|--------|-----------------|
| RAG Pipeline quality | 20% | Chunking, retrieval relevance, prompt design |
| Go API & DB design | 15% | Idiomatic Go, REST semantics, schema quality |
| MCP Server | 15% | Protocol understanding, clean separation |
| Tests | 15% | Meaningful coverage, right boundaries mocked |
| Dockerization & runability | 10% | Single-command startup, seeds data, actually works |
| Error handling | 10% | Explicit, typed, meaningful — no silent failures |
| Design patterns | 10% | Appropriate use with documented reasoning |
| Code quality & documentation | 5% | Clarity, inline comments, API docs, README |
| Fine-tuning (optional) | +10% bonus | Practical understanding of PEFT techniques |

---

## Getting Started

### Forking the Repository

1. Navigate to the challenge repository: `https://github.com/tucows/interview-exercise-ai`
2. Click **Fork** in the top-right corner of the GitHub page to create a copy under your own GitHub account
3. Clone your fork locally:
   ```
   git clone https://github.com/YOUR-USERNAME/interview-exercise-ai.git
   cd interview-exercise-ai
   ```
4. Create a new branch for your work:
   ```
   git checkout -b challenge/your-name
   ```
5. Work exclusively on your branch — do not push directly to `main`

### Submitting Your Solution

When you are ready to submit:

1. Ensure your solution runs cleanly from a single `docker-compose up` on a fresh clone — test this before submitting
2. Push your branch to your fork:
   ```
   git push origin challenge/your-name
   ```
3. Open a **Pull Request** from your fork's branch back to the original repository's `main` branch
4. Title your PR: `[Challenge Submission] Your Full Name`
5. In the PR description, include:
   - Estimated time spent
   - Any setup notes or prerequisites beyond `docker-compose up`
   - Anything you'd like the reviewer to pay particular attention to
   - What you would have done differently or improved with more time
6. Submit via greenhouse link
7. **Submission deadline:** Your completed PR must be submitted within 5 calendar days of receiving this challenge

> If you encounter any issues with the repository setup or have clarifying questions about the requirements, contact youre recruiter. We will not answer questions about implementation choices — those are intentionally left to you.

---

## Rules & Integrity

- You may use any libraries or frameworks you're comfortable with — we're more interested in your decision-making than your choice of HTTP framework
- Local models via Ollama are preferred but not required — if you use a paid API (OpenAI, Anthropic), note the cost implications in your README
- If you run out of time, document what you would have done next — an honest README is valued over incomplete code with no context
- We will do a 30-minute follow-up session to walk through your submission — be prepared to discuss your choices and extend the design live

---

## Engineering Standards

These are not optional — they are evaluated as part of your submission.

### Dockerization
- The entire solution must run with a **single `docker-compose up` command** — no manual setup steps beyond copying an `.env.example`
- Services required: PostgreSQL (with pgvector), the Python RAG service, and the Go API
- All services must be healthy and producing real results when the reviewer runs them
- Include a seed script or `make ingest` command that loads sample documents so the system is immediately queryable

### Error Handling
- All errors must be handled explicitly — no silent failures, bare `except` blocks, or unhandled Go errors
- HTTP responses must use appropriate status codes with structured, consistent error bodies
- Downstream failures (e.g., LLM unavailable, DB connection lost) must be caught and surfaced with meaningful messages
- Error handling strategy should be consistent and deliberate across both services — document your approach in the README

### Design Patterns
Use of design patterns is encouraged where they genuinely improve the design. Document your pattern choices and the reasoning behind them in the README — we want to understand your thinking, not just see the implementation.

### Documentation
Your submission must include:
- **`README.md`** — setup instructions, architecture overview, design decisions, patterns used, known limitations, and what you'd improve with more time. A simple architecture diagram (ASCII or image) is expected.
- **Inline code comments** — explain *why*, not *what*. Non-obvious logic, trade-off choices, and workarounds must be commented.
- **API documentation** — document all endpoints (a Swagger/OpenAPI spec, Postman collection, or well-structured markdown table is acceptable)
- **`schema.sql`** — all tables, indexes, and constraints must have comments explaining their purpose

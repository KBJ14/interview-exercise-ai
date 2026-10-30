# Part 1 Design (RAG pipeline)

Design notes for the Python RAG pipeline as of commit `ae7f4d1`.
How to run it: [run-guide.md](run-guide.md). Evaluation runs: [rag-baseline.md](rag-baseline.md).

## Flow

```
Ingest (scripts/ingest.py -> rag/ingest.py)
  file -> parse_document (loader) -> SHA-256 of raw text
       -> hash already in documents?  yes: skip, no API call
       -> chunk_document (chunker) -> embed_texts (1 batched request)
       -> one transaction: INSERT documents + INSERT embeddings

Question (scripts/ask.py -> rag/generator.py)
  question -> validate (non-empty, 1 <= k <= 20)
           -> embed question -> pgvector top-k by cosine distance (retriever)
           -> no chunks? return "not in documents", no generation call
           -> label chunks S1..Sk -> Responses API with strict JSON schema
           -> validate citations -> answer + cited sources + token usage
```

Modules: `loader` (parse), `chunker` (split + token count), `embedder` (OpenAI embeddings), `db` (connection settings), `ingest` (store), `retriever` (search), `generator` (answer).

## Chunking: one section per chunk, 300 / 50 tokens

- Each section (heading + paragraph) becomes one chunk, prefixed with `<title> > <heading>` so the topic is embedded even when the paragraph does not repeat it.
- A section is split only if it exceeds 300 tokens, at sentence boundaries, with up to 50 tokens of trailing sentences repeated in the next chunk. A sentence longer than the limit is split at word boundaries; a single word or heading over the limit raises an error instead of being cut.
- Why section-based: each section covers one policy, so one chunk answers one kind of question and retrieval returns the whole rule with its exceptions.
- Why 300: current sections are 103–216 tokens, so none is split; 300 leaves room for longer sections while keeping chunks focused.
- Why 50 overlap: only applies inside a split section, where it keeps the sentence that links two halves; between sections there is no overlap because topics differ.
- Tokens are counted with tiktoken `cl100k_base`, the tokenizer of the embedding model, on the final chunk text (header + overlap + body). Local counts matched the API's billed tokens (141 = 141).

## Storage: two tables

| Table | Holds | Key constraints |
|-------|-------|-----------------|
| `documents` | title, source file name, full original text, SHA-256 `content_hash` | `content_hash` UNIQUE (duplicate policy), non-empty checks |
| `embeddings` | chunk text, `vector(1536)`, `embedding_model`, `chunk_index`, `token_count` | FK to `documents` with `ON DELETE CASCADE`, UNIQUE `(document_id, chunk_index, embedding_model)`, optional HNSW cosine index |

- Chunk text and vector share one table, so retrieval is a single join to `documents` for the title.
- The original text is kept so chunks can be rebuilt and re-embedded without the source file.
- Duplicates: the hash lookup runs before any API call. If another run inserts the same content during embedding, `ON CONFLICT DO NOTHING` keeps the existing row and the new embeddings are discarded.
- Rollback: the document row and all chunks are written in one transaction, and no transaction is open while the API is called. Tests force a failure on the last chunk and after all chunks are written, and confirm nothing remains.
- Premise: one embedding model (`text-embedding-3-small`, 1536 dims). Changing models means re-embedding everything; the `embedding_model` column identifies stale rows.

## Models and API cost

| Use | Model | Why |
|-----|-------|-----|
| Embeddings | `text-embedding-3-small`, 1536 dims | No local GPU or model setup on WSL; low cost; 1536 fits pgvector's HNSW limit for `vector` |
| Generation | `gpt-6-luna`, reasoning effort `low` (configurable via `RAG_GENERATION_MODEL`, `RAG_REASONING_EFFORT`) | Smallest current model, recommended by OpenAI for cost-sensitive work; supports strict structured output; short grounded answers do not need more reasoning |

This uses a paid API instead of a local model (the challenge prefers Ollama). Cost is incurred only at these points:

| Action | Requests | Observed size |
|--------|----------|---------------|
| Ingest a new document | 1 embedding request | 631–720 tokens per document |
| Re-ingest identical content | none | skipped before the API |
| `search.py` | 1 embedding request | question length |
| `ask.py` | 1 embedding + at most 1 generation request | about 750–900 tokens per question; generation skipped when nothing is retrieved |
| Tests, `preview_chunks.py` | none | API is mocked or not used |

The 7-question baseline cost about $0.0008 for generation at published standard rates.

## Limitations

- Retrieval always returns the k nearest chunks, even when none is relevant. Distance cannot detect unanswerable questions: the discount question's top distance (0.3857) was closer than some answerable ones. Handling this is left to the prompt and `answerable`.
- Cosine distance is a geometric measure, not a probability or confidence.
- Citation checks cover format only: valid JSON, labels from the supplied sources, inline labels matching the `citations` list, and at least one citation for answerable answers. They do not verify that the answer is true or that a citation supports its sentence.
- Paragraph-level citations are allowed: consecutive sentences from one source may share one label. The code cannot tell whether a final label covers every sentence before it.
- Source text is passed separately from the instructions and HTML-escaped, and the prompt tells the model to ignore instructions inside sources. Whether the model actually ignores injected instructions has not been tested with live calls.
- The baseline covers 7 questions on 15 chunks; results on a larger or more overlapping corpus are unknown.

## Current scope

- The loader parses only the fictional Quillbox format in `data/documents/`: optional `[FICTIONAL ...]` notice, one-line title, then blank-line-separated sections whose first line is the heading. Other layouts raise `DocumentFormatError`.
- Only `.txt` files directly inside the given directory are read. PDF is not supported. Files in `docs/` are never ingested.
- Sentence splitting is a simple punctuation rule; abbreviations such as "e.g." would split incorrectly.
- The pipeline runs as CLI scripts. There is no HTTP service yet, the schema is applied manually, and the Python code is not in `docker-compose.yml`.

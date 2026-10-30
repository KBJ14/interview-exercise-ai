-- Schema for the Part 1 RAG pipeline.
--
-- Premise: a single embedding model is used for every row
--   (text-embedding-3-small, 1536 dimensions).
--   The vector column is fixed at vector(1536) and the HNSW index covers all rows,
--   so mixing models would compare vectors from different spaces.
--   Switching models means re-embedding every chunk (original text is kept in documents.content).

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested source document.
CREATE TABLE documents (
    -- Surrogate key; returned to clients as document_id in query sources.
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Human-readable name shown in listings and answer citations.
    title        TEXT NOT NULL CHECK (btrim(title) <> ''),

    -- Where the document came from (file name, URL). Nullable: text pasted directly has no source.
    source       TEXT,

    -- Full original text, kept so chunks can be rebuilt and re-embedded without the source file.
    content      TEXT NOT NULL CHECK (btrim(content) <> ''),

    -- SHA-256 (hex) of content, computed by the application before insert.
    -- UNIQUE enforces the "identical content is stored only once" policy; the constraint's
    -- index also makes the duplicate check a single lookup.
    -- Only byte-identical content is caught; whitespace or formatting differences count as new documents.
    content_hash TEXT NOT NULL UNIQUE CHECK (content_hash ~ '^[0-9a-f]{64}$'),

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk: the chunk text and its embedding live together (no separate chunks table),
-- which keeps retrieval to a single-table query.
CREATE TABLE embeddings (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- CASCADE: deleting a document removes its chunks, so no orphaned vectors are ever retrieved.
    document_id     BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,

    -- 0-based position of the chunk within its document; lets us restore original order
    -- and show neighbouring context.
    chunk_index     INTEGER NOT NULL CHECK (chunk_index >= 0),

    -- Chunk text that was embedded; this is what gets put into the LLM prompt.
    content         TEXT NOT NULL CHECK (btrim(content) <> ''),

    -- Dimension is enforced by the type: inserting a vector of any other length fails.
    embedding       vector(1536) NOT NULL,

    -- Records which model produced the vector, so stale rows can be found after a model change.
    embedding_model TEXT NOT NULL,

    -- Token count of the chunk (from the embedding tokenizer); used to check chunk sizing
    -- and to budget the prompt.
    token_count     INTEGER NOT NULL CHECK (token_count > 0),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- A document cannot have two chunks at the same position for the same model.
    -- document_id is the leading column, so this index also serves the FK
    -- (cascade deletes and "all chunks of a document" lookups) without a separate index.
    CONSTRAINT embeddings_document_chunk_model_key
        UNIQUE (document_id, chunk_index, embedding_model)
);

-- Optional: speed-up for similarity search, not a requirement for it.
-- Without this index, `ORDER BY embedding <=> $1 LIMIT k` still works as an exact
-- sequential scan, which is fast enough for a small corpus.
-- HNSW is approximate: much faster on large tables, but may occasionally miss a true nearest neighbour.
-- vector_cosine_ops matches the cosine distance operator (<=>); queries must use <=> to use this index.
CREATE INDEX embeddings_embedding_hnsw_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops);

COMMIT;

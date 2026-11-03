-- Schema shared by the Python RAG service and the Go API, which use the same PostgreSQL database.
--   Python writes: rag/ingest.py stores a documents row and all of its embeddings rows in one
--   transaction. Python reads: rag/retriever.py runs the vector search over embeddings.
--   Go reads and deletes: internal/store lists document metadata and deletes documents.
--   Keeping both tables in one database is what makes the single-transaction write and the
--   cascading delete possible; the cost is that a change here must be reflected in both services.
--
-- Premise: a single embedding model is used for every row
--   (text-embedding-3-small, 1536 dimensions).
--   The vector column is fixed at vector(1536) and the HNSW index covers all rows,
--   so mixing models would compare vectors from different spaces.
--   Switching models means re-embedding every chunk (original text is kept in documents.content).
--
-- How the rules below are used:
--   Constraints live in the database rather than in application code, so they hold for both
--   services and for manual SQL alike.
--   NOT NULL marks values a row is useless without; the one nullable column says why.
--   PRIMARY KEY and UNIQUE each create a B-tree index automatically. Their comments name the
--   queries those indexes serve, which is why no separate index is declared for those columns.
--
-- Apply once to an empty database: CREATE statements have no IF NOT EXISTS, so re-running this
-- file on an existing database fails instead of silently leaving an older definition in place.
-- A database created from an earlier version of this file gets later additions from migrations/.

-- All or nothing: if any statement fails, no half-created schema is left behind.
BEGIN;

-- pgvector supplies the vector type, the distance operators (<=> etc.) and the HNSW index method.
-- IF NOT EXISTS: the pgvector image may already have it enabled in this database.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested source document: the metadata Go lists, plus the full original text.
CREATE TABLE documents (
    -- Surrogate key; returned to clients as document_id in query sources.
    -- GENERATED ALWAYS AS IDENTITY: the database assigns each new id, and an INSERT that supplies
    -- its own id is rejected, so the two services can never choose colliding values.
    -- PRIMARY KEY: guarantees id is unique and NOT NULL. Its automatic index serves
    -- DELETE /documents/{id}, the join from embeddings in vector search, and FK checks.
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Human-readable name shown in listings and answer citations.
    -- CHECK: a blank title would appear as an empty entry in listings and citations.
    title        TEXT NOT NULL CHECK (btrim(title) <> ''),

    -- Where the document came from (file name, URL). Nullable: text pasted directly has no source.
    source       TEXT,

    -- Full original text, kept so chunks can be rebuilt and re-embedded without the source file.
    -- CHECK: an empty document has nothing to chunk or embed.
    content      TEXT NOT NULL CHECK (btrim(content) <> ''),

    -- SHA-256 (hex) of content, computed by the application before insert.
    -- UNIQUE enforces the "identical content is stored only once" policy; the constraint's
    -- index also makes the duplicate check a single lookup.
    -- That index also backs ON CONFLICT (content_hash), which turns two concurrent ingests of the
    -- same text into one row instead of an error.
    -- Only byte-identical content is caught; whitespace or formatting differences count as new documents.
    -- CHECK: accepts only 64 lowercase hex characters, so a wrongly formatted hash from the
    -- application is rejected instead of silently defeating the duplicate check.
    -- Stored although derivable from content: a UNIQUE index on a short fixed-size value is far
    -- cheaper than comparing full texts.
    content_hash TEXT NOT NULL UNIQUE CHECK (content_hash ~ '^[0-9a-f]{64}$'),

    -- Insert time; GET /documents lists newest first by (created_at, id).
    -- Indexed together with id by documents_created_at_id_idx below.
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk: the chunk text and its embedding live together (no separate chunks table),
-- which keeps retrieval to a single-table query.
CREATE TABLE embeddings (
    -- Surrogate key for a chunk row; not exposed by the API.
    -- GENERATED ALWAYS AS IDENTITY: assigned by the database; explicit ids are rejected.
    -- PRIMARY KEY: guarantees id is unique and NOT NULL. Its automatic index is not used by the
    -- current queries, which go through document_id or the vector index.
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- CASCADE: deleting a document removes its chunks, so no orphaned vectors are ever retrieved.
    -- FOREIGN KEY: a chunk can never point to a document that does not exist.
    -- NOT NULL: every chunk belongs to a document.
    document_id     BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,

    -- 0-based position of the chunk within its document; lets us restore original order
    -- and show neighbouring context.
    -- CHECK: positions start at 0; a negative value can only come from a bug.
    chunk_index     INTEGER NOT NULL CHECK (chunk_index >= 0),

    -- Chunk text that was embedded; this is what gets put into the LLM prompt.
    -- CHECK: a blank chunk would embed to noise and waste a slot in the prompt.
    content         TEXT NOT NULL CHECK (btrim(content) <> ''),

    -- Dimension is enforced by the type: inserting a vector of any other length fails.
    -- 1536 is the output size of text-embedding-3-small, and stays below pgvector's
    -- 2000-dimension limit for HNSW indexes on the vector type.
    embedding       vector(1536) NOT NULL,

    -- Records which model produced the vector, so stale rows can be found after a model change.
    -- Kept as a plain string rather than a separate models table: under the single-model premise
    -- a lookup table would add a join without removing any real duplication.
    embedding_model TEXT NOT NULL,

    -- Token count of the chunk (from the embedding tokenizer); used to check chunk sizing
    -- and to budget the prompt.
    -- CHECK: a stored chunk always has at least one token.
    token_count     INTEGER NOT NULL CHECK (token_count > 0),

    -- Insert time of the chunk, for diagnosing when a document was (re-)embedded; not queried.
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
-- An index built for another operator (L2 <->, inner product <#>) would be ignored by those queries.
-- Built with pgvector's default parameters. hnsw.ef_search (default 40) sets the size of the
-- candidate list the search keeps while walking the graph: a larger list explores more of it,
-- which raises recall and costs speed. It bounds how many rows one index scan can produce; it is
-- not the number of results, which LIMIT k decides.
-- Model filter limit: rag/retriever.py adds WHERE embedding_model = ..., which is applied to the
-- rows the index scan produces. With one model every row passes. If several models were stored,
-- filtered-out rows are not replaced (iterative scans are off by default), so a search could
-- return fewer than k rows; per-model partial indexes or pgvector's iterative index scans would
-- then be needed.
CREATE INDEX embeddings_embedding_hnsw_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Metadata listing: GET /documents runs ORDER BY created_at DESC, id DESC (internal/store).
-- A B-tree on exactly those columns and directions lets PostgreSQL read rows already in that
-- order instead of sorting them, and id as the second column keeps ties in a stable order.
-- It does not reduce how many rows a full listing reads (every row is still returned).
-- Whether the planner uses it depends on its cost estimates for the data at hand (row count,
-- row width, memory for sorting): a full listing often stays a sequential scan plus sort, while
-- a query that reads only the first rows of this order (e.g. a page with LIMIT) is more likely
-- to use the index and skip the sort.
-- Existing databases get this index from migrations/001_documents_created_at_id_idx.sql.
CREATE INDEX documents_created_at_id_idx
    ON documents (created_at DESC, id DESC);

COMMIT;

"""Semantic search over stored chunks with pgvector cosine distance.

cosine_distance = 1 - cosine similarity: 0 means the same direction, and smaller is closer.
It is a geometric distance between two vectors, not a probability or a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from rag import db
from rag.embedder import EMBEDDING_MODEL, embed_texts
from rag.ingest import to_vector_literal

DEFAULT_K = 3
MAX_K = 20


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: int
    title: str
    chunk_index: int
    content: str
    cosine_distance: float


# <=> is pgvector's cosine distance operator; ORDER BY on it lets the HNSW index serve the query.
# With HNSW the model filter is applied to index candidates, so a corpus mixing many models could
# return fewer than k rows; with one model (the current premise) this cannot happen.
_SEARCH_SQL = """
SELECT e.document_id, d.title, e.chunk_index, e.content,
       e.embedding <=> %(query)s::vector AS cosine_distance
FROM embeddings e
JOIN documents d ON d.id = e.document_id
WHERE e.embedding_model = %(model)s
ORDER BY e.embedding <=> %(query)s::vector
LIMIT %(k)s
"""


def validate_query(question: object, k: object) -> str:
    """Reject bad input before any API call; returns the stripped question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    # bool is a subclass of int, so True would otherwise pass as k=1.
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be an integer between 1 and {MAX_K}, got {k!r}")
    return question.strip()


def search(
    question: str,
    k: int = DEFAULT_K,
    *,
    embedding_model: str = EMBEDDING_MODEL,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    """Return the k chunks closest to `question`, nearest first.

    `embedding_model` and `conn` exist so tests can search isolated rows inside a rolled-back
    transaction; normal callers use the defaults (the embedder's model, a fresh read-only connection).
    """
    question = validate_query(question, k)
    query_vector = embed_texts([question]).vectors[0]
    params = {"query": to_vector_literal(query_vector), "model": embedding_model, "k": k}

    if conn is not None:
        rows = conn.execute(_SEARCH_SQL, params).fetchall()
    else:
        with db.connect(autocommit=True) as own:
            rows = own.execute(_SEARCH_SQL, params).fetchall()
    return [RetrievedChunk(*row) for row in rows]

"""Ingest one document file: hash, de-duplicate, chunk, embed, store.

Order matters for cost and consistency:
1. the duplicate check runs before any API call, so re-ingesting a file is free;
2. embeddings are created with no DB transaction open, so a slow API never holds locks;
3. the document row and all its chunks are written in one transaction (all or nothing).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg

from rag import db
from rag.chunker import Chunk, chunk_document
from rag.embedder import EMBEDDING_MODEL, embed_texts
from rag.loader import parse_document


@dataclass(frozen=True)
class IngestResult:
    status: str  # "inserted" or "skipped"
    document_id: int
    source: str
    chunk_count: int
    api_calls: int
    embedding_tokens: int = 0


def content_hash(text: str) -> str:
    # Hash of the exact text stored in documents.content (see the CHECK in schema.sql).
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_vector_literal(vector: list[float]) -> str:
    # pgvector's text input format; avoids a separate adapter package.
    return "[" + ",".join(repr(x) for x in vector) + "]"


def ingest_file(path: str | Path) -> IngestResult:
    path = Path(path).expanduser().resolve()
    doc = parse_document(path.name, path.read_text(encoding="utf-8"))
    digest = content_hash(doc.raw_text)

    # autocommit: this lookup must not leave a transaction open while we call the API.
    with db.connect(autocommit=True) as conn:
        existing = _find_document(conn, digest)
    if existing is not None:
        return IngestResult("skipped", existing[0], doc.source, existing[1], api_calls=0)

    chunks = chunk_document(doc)
    embedded = embed_texts([c.content for c in chunks])

    with db.connect() as conn, conn.transaction():
        row = conn.execute(
            """
            INSERT INTO documents (title, source, content, content_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id
            """,
            (doc.title, doc.source, doc.raw_text, digest),
        ).fetchone()
        if row is None:
            # Another run stored the same content while we were embedding; keep theirs, discard ours.
            document_id, chunk_count = _find_document(conn, digest)
            return IngestResult("skipped", document_id, doc.source, chunk_count, api_calls=1,
                                embedding_tokens=embedded.total_tokens)
        document_id = row[0]
        _insert_chunks(conn, document_id, chunks, embedded.vectors)

    return IngestResult("inserted", document_id, doc.source, len(chunks), api_calls=1,
                        embedding_tokens=embedded.total_tokens)


def _find_document(conn: psycopg.Connection, digest: str) -> tuple[int, int] | None:
    return conn.execute(
        """
        SELECT d.id, count(e.id)
        FROM documents d LEFT JOIN embeddings e ON e.document_id = d.id
        WHERE d.content_hash = %s
        GROUP BY d.id
        """,
        (digest,),
    ).fetchone()


def _insert_chunks(conn: psycopg.Connection, document_id: int, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO embeddings (document_id, chunk_index, content, embedding, embedding_model, token_count)
            VALUES (%s, %s, %s, %s::vector, %s, %s)
            """,
            [
                (document_id, c.chunk_index, c.content, to_vector_literal(v), EMBEDDING_MODEL, c.token_count)
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )

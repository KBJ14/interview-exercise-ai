import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import psycopg

from rag import db, ingest
from rag.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingResult


def write_doc(directory: str, marker: str) -> Path:
    """A small document in the corpus format; `marker` makes its content (and hash) unique."""
    text = (
        "[FICTIONAL DOCUMENT - ingest test]\n\n"
        f"Ingest Test {marker}\n\n"
        f"First Section\nFirst body for {marker}. It has two sentences.\n\n"
        f"Second Section\nSecond body for {marker}. It also has two sentences.\n\n"
        f"Third Section\nThird body for {marker}. This is the last section.\n"
    )
    path = Path(directory) / f"ingest-test-{marker}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def fake_embeddings(texts: list[str], bad_last: bool = False) -> EmbeddingResult:
    vectors = [[0.01] * EMBEDDING_DIMENSIONS for _ in texts]
    if bad_last:
        vectors[-1] = [0.01] * 3  # wrong size: pgvector rejects this row inside the INSERT
    return EmbeddingResult(model=EMBEDDING_MODEL, vectors=vectors, prompt_tokens=0, total_tokens=0)


class DuplicateSkipTest(unittest.TestCase):
    def test_existing_hash_returns_skipped_without_calling_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_doc(tmp, "dup")
            expected_hash = ingest.content_hash(path.read_text(encoding="utf-8"))
            conn = mock.MagicMock()
            conn.__enter__.return_value = conn
            conn.execute.return_value.fetchone.return_value = (42, 3)  # (document_id, chunk_count)

            with mock.patch.object(ingest.db, "connect", return_value=conn) as connect, \
                 mock.patch.object(ingest, "embed_texts") as embed:
                result = ingest.ingest_file(path)

        embed.assert_not_called()
        self.assertEqual((result.status, result.document_id, result.chunk_count, result.api_calls), ("skipped", 42, 3, 0))
        # Only the read-only lookup connection was opened; no write transaction followed.
        connect.assert_called_once_with(autocommit=True)
        sql, params = conn.execute.call_args.args
        self.assertIn("content_hash", sql)
        self.assertEqual(params, (expected_hash,))


def _db_available() -> bool:
    try:
        with db.connect(connect_timeout=2):
            return True
    except (psycopg.Error, RuntimeError):
        return False


@unittest.skipUnless(_db_available(), "PostgreSQL not reachable (start it with `docker compose up -d`)")
class RollbackOnStoreFailureTest(unittest.TestCase):
    """Runs the real ingest_file write path against the real database; only the embedding API is faked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.marker = uuid.uuid4().hex
        self.path = write_doc(self.tmp.name, self.marker)
        self.digest = ingest.content_hash(self.path.read_text(encoding="utf-8"))
        self.totals_before = self._totals()

    def tearDown(self):
        # Safety net if an assertion failed after a commit: remove only this test's document (chunks cascade).
        with db.connect() as conn:
            conn.execute("DELETE FROM documents WHERE content_hash = %s", (self.digest,))
        self.tmp.cleanup()

    def _totals(self) -> tuple[int, int]:
        with db.connect(autocommit=True) as conn:
            return conn.execute("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM embeddings)").fetchone()

    def _leftovers(self) -> tuple[int, int]:
        """Rows of this test's document, read on a fresh connection so only committed data is visible."""
        with db.connect(autocommit=True) as conn:
            return conn.execute(
                """
                SELECT (SELECT count(*) FROM documents WHERE content_hash = %s),
                       (SELECT count(*) FROM embeddings WHERE content LIKE %s)
                """,
                (self.digest, f"%{self.marker}%"),
            ).fetchone()

    def test_invalid_last_chunk_rolls_back_document_and_earlier_chunks(self):
        with mock.patch.object(ingest, "embed_texts", side_effect=lambda texts: fake_embeddings(texts, bad_last=True)):
            with self.assertRaises(psycopg.errors.DataException) as ctx:
                ingest.ingest_file(self.path)

        self.assertIn(f"expected {EMBEDDING_DIMENSIONS} dimensions", str(ctx.exception))
        self.assertEqual(self._leftovers(), (0, 0))
        self.assertEqual(self._totals(), self.totals_before)

    def test_rows_already_written_in_the_transaction_are_rolled_back(self):
        real_insert = ingest._insert_chunks
        seen_inside = {}

        def insert_then_fail(conn, document_id, chunks, vectors):
            real_insert(conn, document_id, chunks, vectors)
            # Same connection, same open transaction: the document and every chunk exist here.
            seen_inside["rows"] = conn.execute(
                "SELECT (SELECT count(*) FROM documents WHERE id = %s), (SELECT count(*) FROM embeddings WHERE document_id = %s)",
                (document_id, document_id),
            ).fetchone()
            raise RuntimeError("simulated failure after storing chunks")

        with mock.patch.object(ingest, "embed_texts", side_effect=fake_embeddings), \
             mock.patch.object(ingest, "_insert_chunks", side_effect=insert_then_fail):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                ingest.ingest_file(self.path)

        self.assertEqual(seen_inside["rows"], (1, 3))
        self.assertEqual(self._leftovers(), (0, 0))
        self.assertEqual(self._totals(), self.totals_before)


if __name__ == "__main__":
    unittest.main()

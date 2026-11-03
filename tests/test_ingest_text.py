"""ingest_text: document formats, duplicates, concurrency, rollback.

Embeddings are always faked (no OpenAI calls). Database tests write committed rows only for
documents whose content carries a random marker, and delete exactly those rows in tearDown;
the existing corpus is checked to be unchanged.
"""

import unittest
import uuid
from unittest import mock

import psycopg

from rag import db, ingest
from rag.chunker import Chunk, chunk_document
from rag.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingResult
from rag.loader import DocumentFormatError


def fake_embeddings(texts):
    return EmbeddingResult(model=EMBEDDING_MODEL, vectors=[[0.01] * EMBEDDING_DIMENSIONS for _ in texts],
                           prompt_tokens=0, total_tokens=len(texts))


def sectioned_text(marker: str) -> str:
    return (f"[FICTIONAL DOCUMENT - ingest_text test]\n\nIngest Text Test {marker}\n\n"
            f"First Section\nFirst body for {marker}.\n\nSecond Section\nSecond body for {marker}.\n")


class BuildDocumentTest(unittest.TestCase):
    """Format rules; no database or API involved."""

    def test_plain_text_is_one_untitled_section_and_chunks_under_the_title(self):
        doc = ingest.build_document("First line.\n\nSecond   paragraph.\n", source="notes.txt", title="My Notes", fmt="plain")
        self.assertEqual((doc.title, doc.source, len(doc.sections)), ("My Notes", "notes.txt", 1))
        self.assertEqual(doc.sections[0].body, "First line. Second paragraph.")
        self.assertEqual(chunk_document(doc)[0].content, "My Notes\nFirst line. Second paragraph.")

    def test_plain_title_falls_back_to_source_stem_without_touching_the_filesystem(self):
        doc = ingest.build_document("Body.", source="../../etc/returns-policy.txt", title=None, fmt="plain")
        self.assertEqual(doc.title, "returns-policy")

    def test_plain_without_title_or_source_is_rejected(self):
        with self.assertRaises(DocumentFormatError):
            ingest.build_document("Body.", source=None, title=None, fmt="plain")

    def test_sectioned_uses_the_existing_parser_and_keeps_source(self):
        doc = ingest.build_document(sectioned_text("m"), source=None, title=None, fmt="sectioned")
        self.assertEqual((doc.title, doc.source, [s.heading for s in doc.sections]),
                         ("Ingest Text Test m", None, ["First Section", "Second Section"]))

    def test_sectioned_rejects_title_override_and_wrong_layout(self):
        with self.assertRaises(DocumentFormatError):
            ingest.build_document(sectioned_text("m"), source=None, title="Other", fmt="sectioned")
        with self.assertRaises(DocumentFormatError):
            ingest.build_document("Just one paragraph of plain text.", source="a.txt", title=None, fmt="sectioned")

    def test_nul_and_unknown_format_are_rejected(self):
        with self.assertRaises(DocumentFormatError):
            ingest.build_document("bad\x00text", source="a.txt", title="T", fmt="plain")
        with self.assertRaises(DocumentFormatError):
            ingest.build_document("text", source="a.txt", title="T", fmt="markdown")


def _db_available() -> bool:
    try:
        with db.connect(connect_timeout=2):
            return True
    except (psycopg.Error, RuntimeError):
        return False


@unittest.skipUnless(_db_available(), "PostgreSQL not reachable (start it with `docker compose up -d`)")
class IngestTextDatabaseTest(unittest.TestCase):
    def setUp(self):
        self.marker = f"ingest-text-test-{uuid.uuid4().hex}"
        self.texts = []  # every text this test may have stored; cleaned up by hash
        self.before = self._corpus()

    def tearDown(self):
        with db.connect() as conn:
            for text in self.texts:
                conn.execute("DELETE FROM documents WHERE content_hash = %s", (ingest.content_hash(text),))
        self.assertEqual(self._corpus(), self.before, "pre-existing documents or embeddings changed")

    def _corpus(self):
        with db.connect(autocommit=True) as conn:
            return conn.execute(
                "SELECT (SELECT string_agg(id || ':' || content_hash, ',' ORDER BY id) FROM documents"
                "        WHERE content NOT LIKE '%%ingest-text-test-%%'),"
                "       (SELECT count(*) FROM embeddings e JOIN documents d ON d.id = e.document_id"
                "        WHERE d.content NOT LIKE '%%ingest-text-test-%%')").fetchone()

    def _rows(self, text):
        with db.connect(autocommit=True) as conn:
            return conn.execute(
                "SELECT d.title, d.source, count(e.id) FROM documents d LEFT JOIN embeddings e ON e.document_id = d.id"
                " WHERE d.content_hash = %s GROUP BY d.id", (ingest.content_hash(text),)).fetchall()

    def plain(self, suffix=""):
        text = f"Plain test document {self.marker}{suffix}. It has two sentences."
        self.texts.append(text)
        return text

    def test_plain_text_is_stored_with_its_chunks(self):
        text = self.plain()
        with mock.patch.object(ingest, "embed_texts", side_effect=fake_embeddings) as embed:
            result = ingest.ingest_text(text, source=None, title="Plain Test", fmt="plain")
        embed.assert_called_once()
        self.assertEqual((result.status, result.chunk_count, result.api_calls), ("inserted", 1, 1))
        self.assertEqual(self._rows(text), [("Plain Test", None, 1)])

    def test_duplicate_is_skipped_before_embedding_and_reports_the_existing_row(self):
        text = self.plain()
        with mock.patch.object(ingest, "embed_texts", side_effect=fake_embeddings):
            first = ingest.ingest_text(text, source="a.txt", title="First", fmt="plain")
        with mock.patch.object(ingest, "embed_texts") as embed:
            # Same text with different metadata and format is still the same content.
            second = ingest.ingest_text(text, source="b.txt", title="Second", fmt="plain")
        embed.assert_not_called()
        self.assertEqual((second.status, second.document_id, second.chunk_count, second.api_calls),
                         ("skipped", first.document_id, first.chunk_count, 0))
        self.assertEqual(self._rows(text), [("First", "a.txt", 1)])

    def test_sectioned_text_matches_ingest_file_behaviour(self):
        text = sectioned_text(self.marker)
        self.texts.append(text)
        with mock.patch.object(ingest, "embed_texts", side_effect=fake_embeddings):
            result = ingest.ingest_text(text, source="test.txt", fmt="sectioned")
        self.assertEqual((result.status, result.chunk_count), ("inserted", 2))
        self.assertEqual(self._rows(text), [(f"Ingest Text Test {self.marker}", "test.txt", 2)])

    def test_concurrent_duplicate_during_embedding_is_skipped_via_content_hash_conflict(self):
        text = self.plain()

        def embed_while_another_run_commits(texts):
            # Simulates a second request that stored the same text after our duplicate check.
            with db.connect() as other:
                other.execute("INSERT INTO documents (title, source, content, content_hash) VALUES (%s, %s, %s, %s)",
                              ("Other Run", "other.txt", text, ingest.content_hash(text)))
            return fake_embeddings(texts)

        with mock.patch.object(ingest, "embed_texts", side_effect=embed_while_another_run_commits):
            result = ingest.ingest_text(text, source="mine.txt", title="Mine", fmt="plain")
        self.assertEqual((result.status, result.api_calls), ("skipped", 1))
        self.assertEqual(self._rows(text), [("Other Run", "other.txt", 0)])  # theirs kept, ours discarded

    def test_other_unique_violation_is_not_treated_as_duplicate_and_rolls_back(self):
        text = self.plain()
        doc_chunk = Chunk("t", 0, "", f"Dup Test\n{text}", 5)
        with mock.patch.object(ingest, "embed_texts", side_effect=fake_embeddings), \
             mock.patch.object(ingest, "chunk_document", return_value=[doc_chunk, doc_chunk]):  # same chunk_index twice
            with self.assertRaises(psycopg.errors.UniqueViolation) as ctx:
                ingest.ingest_text(text, source=None, title="Dup Test", fmt="plain")
        self.assertIn("embeddings_document_chunk_model_key", str(ctx.exception))
        self.assertEqual(self._rows(text), [])  # document row rolled back with the chunks

    def test_storage_failure_rolls_back_document_and_chunks(self):
        text = self.plain(" " + " ".join(f"Extra sentence {i} for splitting." for i in range(120)))

        def bad_last_vector(texts):
            result = fake_embeddings(texts)
            result.vectors[-1] = [0.01] * 3
            return result

        with mock.patch.object(ingest, "embed_texts", side_effect=bad_last_vector):
            with self.assertRaises(psycopg.errors.DataException):
                ingest.ingest_text(text, source=None, title="Rollback Test", fmt="plain")
        self.assertEqual(self._rows(text), [])


if __name__ == "__main__":
    unittest.main()

import math
import unittest
import uuid
from unittest import mock

import psycopg

from rag import db, retriever
from rag.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingResult
from rag.ingest import to_vector_literal


def basis(*indices: int) -> list[float]:
    """Vector with 1.0 at the given positions, so cosine distances are easy to predict."""
    v = [0.0] * EMBEDDING_DIMENSIONS
    for i in indices:
        v[i] = 1.0
    return v


def fake_query_embedding(vector: list[float]):
    return lambda texts: EmbeddingResult(model=EMBEDDING_MODEL, vectors=[vector], prompt_tokens=0, total_tokens=0)


class ValidationTest(unittest.TestCase):
    def test_bad_input_is_rejected_before_calling_embeddings(self):
        cases = [("", 3), ("   \n", 3), (None, 3), ("ok", 0), ("ok", -1), ("ok", retriever.MAX_K + 1),
                 ("ok", 2.5), ("ok", "3"), ("ok", True)]
        for question, k in cases:
            with self.subTest(question=question, k=k), mock.patch.object(retriever, "embed_texts") as embed:
                with self.assertRaises(ValueError):
                    retriever.search(question, k)
                embed.assert_not_called()

    def test_question_is_stripped_before_embedding(self):
        with mock.patch.object(retriever, "embed_texts", side_effect=fake_query_embedding(basis(0))) as embed, \
             mock.patch.object(retriever.db, "connect") as connect:
            connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = []
            retriever.search("  refund?  ", 3)
        embed.assert_called_once_with(["refund?"])


def _db_available() -> bool:
    try:
        with db.connect(connect_timeout=2):
            return True
    except (psycopg.Error, RuntimeError):
        return False


@unittest.skipUnless(_db_available(), "PostgreSQL not reachable (start it with `docker compose up -d`)")
class SearchQueryTest(unittest.TestCase):
    """Runs the real SQL on rows inserted inside a transaction that is always rolled back."""

    def setUp(self):
        self.conn = db.connect()
        self.model = f"test-model-{uuid.uuid4().hex}"  # isolates these rows from the real corpus
        doc_id = self.conn.execute(
            "INSERT INTO documents (title, content, content_hash) VALUES (%s, %s, %s) RETURNING id",
            ("Search Test Doc", "search test", uuid.uuid4().hex + uuid.uuid4().hex),
        ).fetchone()[0]
        self.doc_id = doc_id
        rows = [
            (0, "same direction", basis(0), self.model),
            (1, "45 degrees", basis(0, 1), self.model),
            (2, "orthogonal", basis(1), self.model),
            (3, "same direction, other model", basis(0), self.model + "-other"),
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embeddings (document_id, chunk_index, content, embedding, embedding_model, token_count) "
                "VALUES (%s, %s, %s, %s::vector, %s, 1)",
                [(doc_id, i, text, to_vector_literal(v), model) for i, text, v, model in rows],
            )

    def tearDown(self):
        self.conn.rollback()
        self.conn.close()

    def _search(self, k: int):
        with mock.patch.object(retriever, "embed_texts", side_effect=fake_query_embedding(basis(0))):
            return retriever.search("any question", k, embedding_model=self.model, conn=self.conn)

    def test_returns_nearest_first_with_cosine_distance(self):
        results = self._search(3)
        self.assertEqual([r.chunk_index for r in results], [0, 1, 2])
        expected = [0.0, 1 - 1 / math.sqrt(2), 1.0]
        for r, d in zip(results, expected):
            self.assertAlmostEqual(r.cosine_distance, d, places=5)
        self.assertEqual(results[0].document_id, self.doc_id)
        self.assertEqual(results[0].title, "Search Test Doc")
        self.assertEqual(results[0].content, "same direction")

    def test_k_limits_the_number_of_results(self):
        self.assertEqual([r.chunk_index for r in self._search(2)], [0, 1])

    def test_rows_from_other_embedding_models_are_excluded(self):
        results = self._search(retriever.MAX_K)
        self.assertEqual(len(results), 3)
        self.assertNotIn("same direction, other model", [r.content for r in results])


if __name__ == "__main__":
    unittest.main()

"""Full query flow on a real PostgreSQL database, with only OpenAI replaced.

Real code under test: POST /internal/query (FastAPI) -> generator.answer() -> retriever.search()
-> pgvector SQL with the documents join -> prompt building -> response parsing and citation
checks -> API serialization. Documents are stored with the real ingest_text().

Replaced: the two OpenAI boundaries only. Embeddings come from fixed vectors, and generation
from a fake Responses client that records the prompt it receives. Constructing a real OpenAI
client fails the test, and the run removes OPENAI_API_KEY from the environment.

Isolation: every test gets its own PostgreSQL schema, created from the real schema.sql and
dropped afterwards (also when the test fails). rag/db.py already prefers DATABASE_URL, so the
test points it at the same database with `search_path=<schema>,public`; no application code
knows about the test. The schema holds only the test's documents, so ranking does not depend on
the real corpus, and the existing documents and chunks are checked to be unchanged.
"""

import json
import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import psycopg
from fastapi.testclient import TestClient
from psycopg import sql

from rag import api, db, generator, ingest, retriever
from rag.embedder import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingResult

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema.sql"
QUESTION = "When does the test ferry leave Pier 9?"


def basis(*indices: int) -> list[float]:
    """Unit-length vector over the given axes, so cosine distances are known exactly."""
    v = [0.0] * EMBEDDING_DIMENSIONS
    for i in indices:
        v[i] = 1.0 / len(indices) ** 0.5
    return v


def embeddings(vector: list[float]):
    return lambda texts: EmbeddingResult(EMBEDDING_MODEL, [vector for _ in texts], prompt_tokens=0, total_tokens=0)


class FakeResponsesClient:
    """Stands in for openai.OpenAI: records each responses.create() call, answers citing S1."""

    def __init__(self, answer_text: str):
        self.calls: list[dict] = []
        self._output = json.dumps({"answerable": True, "answer": answer_text, "citations": ["S1"]})
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            status="completed", incomplete_details=None, output_text=self._output, model="fake-generation-model",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120,
                                  output_tokens_details=SimpleNamespace(reasoning_tokens=0)),
        )


def _db_available() -> bool:
    try:
        with db.connect(connect_timeout=2):
            return True
    except (psycopg.Error, RuntimeError):
        return False


def _corpus_fingerprint() -> tuple:
    """Ids, hashes and chunk fingerprints of the real public tables."""
    with db.connect(autocommit=True) as conn:
        return conn.execute(
            """
            SELECT (SELECT coalesce(string_agg(id || ':' || content_hash, ',' ORDER BY id), '') FROM public.documents),
                   (SELECT count(*) FROM public.embeddings),
                   (SELECT coalesce(md5(string_agg(id || ':' || md5(content) || ':' || md5(embedding::text), ',' ORDER BY id)), '')
                    FROM public.embeddings)
            """
        ).fetchone()


@unittest.skipUnless(_db_available(), "PostgreSQL not reachable (start it with `docker compose up -d`)")
class QueryFlowTest(unittest.TestCase):
    def setUp(self):
        self.corpus_before = _corpus_fingerprint()
        self.schema = f"test_query_flow_{uuid.uuid4().hex[:12]}"
        base_url = db.database_url()

        with db.connect(autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.addCleanup(self._drop_schema, base_url)  # registered first, so it runs even if setup fails later

        # Every connection the application opens now resolves unqualified names in the test schema;
        # public stays on the path for the pgvector type and operators.
        scoped_url = psycopg.conninfo.make_conninfo(base_url, options=f"-c search_path={self.schema},public")
        env = mock.patch.dict(os.environ, {"DATABASE_URL": scoped_url})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("OPENAI_API_KEY", None)  # restored by env.stop()

        with db.connect(autocommit=True) as conn:  # the real schema, into the test schema
            conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))

        # Any attempt to build a real OpenAI client fails the test.
        no_openai = mock.patch("rag.embedder.OpenAI", side_effect=AssertionError("real OpenAI client constructed"))
        no_openai.start()
        self.addCleanup(no_openai.stop)

    def _drop_schema(self, base_url):
        with psycopg.connect(base_url, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema)))
        # Checked here so it holds on success and on failure alike.
        with psycopg.connect(base_url, autocommit=True) as conn:
            left = conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname = %s", (self.schema,)).fetchone()[0]
        self.assertEqual(left, 0, f"test schema {self.schema} was not dropped")
        self.assertEqual(_corpus_fingerprint(), self.corpus_before, "real documents or embeddings changed")

    def _ingest(self, title: str, text: str, vector: list[float]):
        with mock.patch.object(ingest, "embed_texts", side_effect=embeddings(vector)):
            result = ingest.ingest_text(text, source=f"{title.lower().replace(' ', '-')}.txt", title=title, fmt="plain")
        self.assertEqual((result.status, result.chunk_count), ("inserted", 1))
        return result.document_id

    def _stored_chunk(self, document_id: int) -> str:
        with db.connect(autocommit=True) as conn:
            return conn.execute("SELECT content FROM embeddings WHERE document_id = %s", (document_id,)).fetchone()[0]

    def test_query_flow_from_http_request_to_cited_answer(self):
        # Three documents at known cosine distances from the question vector basis(0).
        ferry = self._ingest("Test Ferry Timetable", "The test ferry leaves Pier 9 every day at 07:45. The crossing takes 52 minutes.", basis(0))
        tickets = self._ingest("Test Ticket Prices", "Adult tickets for the test ferry cost 14 euros.", basis(0, 1))
        weather = self._ingest("Test Weather Notes", "Fog sometimes delays morning departures.", basis(1))
        ferry_chunk = self._stored_chunk(ferry).split("\n", 1)[1]  # body as stored, without the title header line

        fake_llm = FakeResponsesClient("The test ferry leaves Pier 9 every day at 07:45. [S1]")
        with mock.patch.object(retriever, "embed_texts", side_effect=embeddings(basis(0))) as embed_question, \
             mock.patch.object(generator, "make_client", return_value=fake_llm):
            response = TestClient(api.app).post("/internal/query", json={"question": f"  {QUESTION}  ", "k": 3})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        # Answer: produced by the real parsing and citation checks from the fake model output.
        self.assertTrue(body["answerable"])
        self.assertEqual(body["answer"], "The test ferry leaves Pier 9 every day at 07:45. [S1]")
        self.assertEqual(body["citations"], ["S1"])
        self.assertEqual(body["model"], "fake-generation-model")
        embed_question.assert_called_once_with([QUESTION])  # stripped question, embedded once

        # Sources: real pgvector ranking and the documents join, in nearest-first order.
        got = [(s["label"], s["document_id"], s["title"], s["cited"]) for s in body["sources"]]
        self.assertEqual(got, [
            ("S1", ferry, "Test Ferry Timetable", True),
            ("S2", tickets, "Test Ticket Prices", False),
            ("S3", weather, "Test Weather Notes", False),
        ])
        for source, expected in zip(body["sources"], [0.0, 1 - 2 ** -0.5, 1.0]):
            self.assertAlmostEqual(source["cosine_distance"], expected, places=5)
        s1 = body["sources"][0]
        self.assertEqual(s1["chunk"], ferry_chunk)
        self.assertEqual(s1["chunk"], "The test ferry leaves Pier 9 every day at 07:45. The crossing takes 52 minutes.")
        self.assertEqual(s1["chunk_index"], 0)

        # Prompt: the real build_input() received the chunk text read from the database.
        self.assertEqual(len(fake_llm.calls), 1)
        prompt = fake_llm.calls[0]["input"][0]["content"]
        self.assertIn(f'<source id="S1" document="Test Ferry Timetable"', prompt)
        self.assertIn(ferry_chunk, prompt)
        self.assertIn(QUESTION, prompt)
        self.assertNotIn(ferry_chunk, fake_llm.calls[0]["instructions"])  # sources stay out of the rules
        self.assertEqual(
            fake_llm.calls[0]["text"]["format"]["schema"]["properties"]["citations"]["items"]["enum"],
            ["S1", "S2", "S3"],
        )

    def test_citation_of_an_unretrieved_source_is_rejected_end_to_end(self):
        # Only one document exists, so S2 is not a valid label; the real checks must refuse the answer.
        self._ingest("Test Ferry Timetable", "The test ferry leaves Pier 9 every day at 07:45.", basis(0))
        fake_llm = FakeResponsesClient("It leaves at 07:45. [S2]")
        fake_llm._output = json.dumps({"answerable": True, "answer": "It leaves at 07:45. [S2]", "citations": ["S2"]})
        with mock.patch.object(retriever, "embed_texts", side_effect=embeddings(basis(0))), \
             mock.patch.object(generator, "make_client", return_value=fake_llm):
            response = TestClient(api.app).post("/internal/query", json={"question": QUESTION})

        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["error"]["code"], "llm_invalid_response")
        self.assertEqual(len(fake_llm.calls), 1)


if __name__ == "__main__":
    unittest.main()

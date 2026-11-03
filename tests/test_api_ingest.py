import json
import unittest
from unittest import mock

import openai
import psycopg
from fastapi.testclient import TestClient

from rag import api
from rag.chunker import ChunkingError
from rag.ingest import IngestResult
from rag.loader import DocumentFormatError

INSERTED = IngestResult("inserted", 41, "notes.txt", 2, api_calls=1, embedding_tokens=120)
SKIPPED = IngestResult("skipped", 3, "billing.txt", 5, api_calls=0)


class IngestApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def post(self, body, result=INSERTED, error=None, raw: bytes | None = None, headers=None):
        with mock.patch.object(api, "ingest_text", return_value=result, side_effect=error) as fake:
            if raw is not None:
                response = self.client.post("/internal/ingest", content=raw,
                                            headers=headers or {"Content-Type": "application/json"})
            else:
                response = self.client.post("/internal/ingest", json=body)
        return response, fake

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(set(body), {"error"})
        self.assertEqual(body["error"]["code"], code)
        self.assertTrue(body["error"]["message"])
        return body["error"]["message"]


class IngestSuccessTest(IngestApiTestCase):
    def test_new_document_is_201_with_id_status_and_chunk_count(self):
        response, fake = self.post({"text": "Hello world.", "source": "notes.txt", "title": "Notes", "format": "plain"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"status": "inserted", "document_id": 41, "chunk_count": 2})
        fake.assert_called_once_with("Hello world.", source="notes.txt", title="Notes", fmt="plain")

    def test_duplicate_is_200_skipped_with_existing_document(self):
        response, _ = self.post({"text": "same text", "format": "sectioned"}, result=SKIPPED)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "skipped", "document_id": 3, "chunk_count": 5})

    def test_format_defaults_to_plain_and_optional_fields_to_none(self):
        _, fake = self.post({"text": "Hello world."})
        fake.assert_called_once_with("Hello world.", source=None, title=None, fmt="plain")


class IngestInputErrorTest(IngestApiTestCase):
    def test_invalid_fields_are_400_and_never_ingest(self):
        cases = [
            {}, {"text": None}, {"text": ""}, {"text": "  \n "}, {"text": 42}, {"text": ["a"]},
            {"text": "ok", "format": "pdf"}, {"text": "ok", "format": None},
            {"text": "ok", "source": ""}, {"text": "ok", "source": 7}, {"text": "ok", "source": "a" * 256},
            {"text": "ok", "title": "   "}, {"text": "ok", "title": "t" * 301},
            {"text": "ok", "path": "/etc/passwd"},
        ]
        for body in cases:
            with self.subTest(body=body):
                response, fake = self.post(body)
                self.assert_error(response, 400, "invalid_request")
                fake.assert_not_called()

    def test_body_that_is_not_a_json_object_is_400(self):
        for raw in (b"{not json", b"[]", b'"text"', b"\xff\xfe"):
            with self.subTest(raw=raw):
                response, fake = self.post(None, raw=raw)
                self.assert_error(response, 400, "invalid_request")
                fake.assert_not_called()

    def test_document_errors_from_ingest_are_400_invalid_document(self):
        for error in (DocumentFormatError("text: expected a title block followed by at least one section"),
                      DocumentFormatError("document contains a NUL character"),
                      ChunkingError("a single word exceeds the chunk limit")):
            with self.subTest(error=str(error)):
                response, _ = self.post({"text": "x"}, error=error)
                self.assertEqual(self.assert_error(response, 400, "invalid_document"), str(error))


class IngestSizeLimitTest(IngestApiTestCase):
    def test_document_over_limit_is_413_and_never_ingests(self):
        response, fake = self.post({"text": "a" * (api.MAX_DOCUMENT_BYTES + 1)})
        self.assert_error(response, 413, "document_too_large")
        fake.assert_not_called()

    def test_limit_counts_utf8_bytes_not_characters(self):
        text = "가" * (api.MAX_DOCUMENT_BYTES // 3 + 1)  # 3 bytes each in UTF-8
        response, fake = self.post({"text": text})
        self.assert_error(response, 413, "document_too_large")
        fake.assert_not_called()

    def test_document_at_limit_is_accepted(self):
        response, fake = self.post({"text": "a" * api.MAX_DOCUMENT_BYTES})
        self.assertEqual(response.status_code, 201, response.text)
        fake.assert_called_once()

    def test_request_body_over_limit_is_413_before_parsing(self):
        raw = json.dumps({"text": "a" * (api.MAX_INGEST_BODY_BYTES + 10)}).encode()
        response, fake = self.post(None, raw=raw)
        self.assert_error(response, 413, "request_too_large")
        fake.assert_not_called()


class IngestFailureTest(IngestApiTestCase):
    def test_storage_and_upstream_failures_reuse_the_existing_classification(self):
        cases = [
            (psycopg.OperationalError("connection refused"), 503, "database_unavailable"),
            # A constraint other than the content_hash duplicate must surface, not become "skipped".
            (psycopg.errors.UniqueViolation("duplicate key value violates unique constraint "
                                            "embeddings_document_chunk_model_key"), 500, "internal_error"),
            (openai.APIConnectionError(request=mock.Mock()), 503, "llm_unavailable"),
            (KeyError("bug"), 500, "internal_error"),
        ]
        for error, status, code in cases:
            with self.subTest(error=type(error).__name__):
                with self.assertLogs("rag.api", level="ERROR") if status == 500 else _nothing():
                    response, _ = self.post({"text": "x"}, error=error)
                message = self.assert_error(response, status, code)
                self.assertNotIn("embeddings_document_chunk_model_key", message)


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    unittest.main()

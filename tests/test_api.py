import unittest
from unittest import mock

import httpx2
import openai
import psycopg
from fastapi.testclient import TestClient

from rag import api
from rag.embedder import ConfigurationError, EmbeddingError
from rag.generator import AnswerResult, GenerationError, Source, Usage
from rag.retriever import MAX_K

ANSWER = AnswerResult(
    question="Can I get a refund for a monthly renewal charge?",
    answerable=True,
    answer="Monthly renewals are not refundable. [S1]",
    citations=["S1"],
    sources=[
        Source("S1", 3, "Quillbox Billing", "Refund Policy", 3, 0.3775, "Monthly renewals are not refundable."),
        Source("S2", 3, "Quillbox Billing", "Failed Payments", 2, 0.4953, "Retries happen 3 times."),
    ],
    model="gpt-6-luna",
    usage=Usage(723, 58, 0, 781),
)

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def status_error(cls, status: int):
    return cls("upstream said no", response=httpx2.Response(status, request=_REQUEST), body=None)


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        # raise_server_exceptions=False: see the 500 response a real client would get.
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def post_query(self, body, answer_return=ANSWER, answer_error=None):
        with mock.patch.object(api, "answer", return_value=answer_return, side_effect=answer_error) as fake:
            response = self.client.post("/internal/query", json=body)
        return response, fake

    def assert_error(self, response, status: int, code: str):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(set(body), {"error"})
        self.assertEqual(set(body["error"]), {"code", "message"})
        self.assertEqual(body["error"]["code"], code)
        self.assertTrue(body["error"]["message"])
        return body["error"]["message"]


class HealthTest(ApiTestCase):
    def test_health_is_ok_without_touching_db_or_openai(self):
        with mock.patch.object(api, "answer") as fake_answer, mock.patch("rag.db.connect") as fake_connect:
            response = self.client.get("/health")
        self.assertEqual((response.status_code, response.json()), (200, {"status": "ok"}))
        fake_answer.assert_not_called()
        fake_connect.assert_not_called()


class QuerySuccessTest(ApiTestCase):
    def test_returns_answer_sources_and_usage(self):
        response, fake = self.post_query({"question": "  Can I get a refund for a monthly renewal charge?  ", "k": 2})
        self.assertEqual(response.status_code, 200)
        fake.assert_called_once_with("Can I get a refund for a monthly renewal charge?", 2)
        body = response.json()
        self.assertEqual((body["answerable"], body["citations"], body["model"]), (True, ["S1"], "gpt-6-luna"))
        self.assertEqual(body["sources"][0], {
            "label": "S1", "document_id": 3, "title": "Quillbox Billing", "section": "Refund Policy",
            "chunk_index": 3, "chunk": "Monthly renewals are not refundable.", "cosine_distance": 0.3775, "cited": True,
        })
        self.assertFalse(body["sources"][1]["cited"])
        self.assertEqual(body["usage"], {"input_tokens": 723, "output_tokens": 58, "reasoning_tokens": 0, "total_tokens": 781})

    def test_k_defaults_to_three(self):
        _, fake = self.post_query({"question": "refund?"})
        fake.assert_called_once_with("refund?", 3)

    def test_unanswerable_result_without_model_call_has_null_usage(self):
        no_info = AnswerResult("q", False, "The provided documents do not contain this information.", [], [], None, None)
        response, _ = self.post_query({"question": "q"}, answer_return=no_info)
        body = response.json()
        self.assertEqual((response.status_code, body["answerable"], body["sources"], body["usage"]), (200, False, [], None))


class QueryInputErrorTest(ApiTestCase):
    def test_invalid_question_or_k_is_400_and_never_calls_answer(self):
        cases = [
            {"question": ""}, {"question": "   "}, {"question": None}, {}, {"question": 42},
            {"question": "ok", "k": 0}, {"question": "ok", "k": MAX_K + 1}, {"question": "ok", "k": 2.5},
            {"question": "ok", "k": "3"}, {"question": "ok", "k": True},
            {"question": "ok", "extra": 1},
        ]
        for body in cases:
            with self.subTest(body=body):
                response, fake = self.post_query(body)
                self.assert_error(response, 400, "invalid_request")
                fake.assert_not_called()

    def test_malformed_json_is_400(self):
        with mock.patch.object(api, "answer") as fake:
            response = self.client.post("/internal/query", content=b"{not json", headers={"Content-Type": "application/json"})
        self.assert_error(response, 400, "invalid_request")
        fake.assert_not_called()

    def test_unknown_route_and_wrong_method_use_the_same_error_shape(self):
        self.assert_error(self.client.get("/nope"), 404, "not_found")
        self.assert_error(self.client.get("/internal/query"), 405, "method_not_allowed")


class QueryFailureTest(ApiTestCase):
    def test_failures_map_to_status_by_meaning(self):
        cases = [
            # (exception from answer(), status, code)
            (GenerationError("response status 'incomplete' (reason: max_output_tokens)"), 502, "llm_invalid_response"),
            (GenerationError("citations to sources that were not provided: ['S4']"), 502, "llm_invalid_response"),
            (EmbeddingError("expected 1536 dimensions, got [3]"), 502, "embedding_invalid_response"),
            (ConfigurationError("OPENAI_API_KEY is not set"), 500, "configuration_error"),
            (openai.APITimeoutError(request=_REQUEST), 504, "llm_timeout"),
            (openai.APIConnectionError(request=_REQUEST), 503, "llm_unavailable"),
            (status_error(openai.RateLimitError, 429), 503, "llm_rate_limited"),
            (status_error(openai.InternalServerError, 500), 503, "llm_unavailable"),
            (status_error(openai.AuthenticationError, 401), 502, "llm_request_rejected"),
            (status_error(openai.NotFoundError, 404), 502, "llm_request_rejected"),
            (psycopg.OperationalError("connection refused"), 503, "database_unavailable"),
            (ValueError("bug after validation"), 500, "internal_error"),
            (KeyError("anything else"), 500, "internal_error"),
        ]
        for error, status, code in cases:
            with self.subTest(error=type(error).__name__, code=code):
                with self.assertLogs("rag.api", level="ERROR") if status == 500 or code == "llm_request_rejected" else _nothing():
                    response, _ = self.post_query({"question": "refund?"}, answer_error=error)
                self.assert_error(response, status, code)

    def test_upstream_and_internal_details_are_not_leaked(self):
        auth = openai.AuthenticationError("Incorrect API key provided: sk-proj-****abcd",
                                          response=httpx2.Response(401, request=_REQUEST), body=None)
        with self.assertLogs("rag.api", level="ERROR"):
            response, _ = self.post_query({"question": "refund?"}, answer_error=auth)
        self.assertNotIn("sk-", self.assert_error(response, 502, "llm_request_rejected"))

        with self.assertLogs("rag.api", level="ERROR"):
            response, _ = self.post_query({"question": "refund?"}, answer_error=KeyError("internal detail"))
        self.assertEqual(self.assert_error(response, 500, "internal_error"), "unexpected internal error")


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    unittest.main()

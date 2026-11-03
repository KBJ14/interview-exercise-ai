"""MCP server tests. answer() is always mocked: no database or OpenAI calls.

In-memory tests use the SDK's Client (initialize -> tools/list -> tools/call). The stdio tests
start `python -m rag.mcp_server` as a subprocess, the way an MCP host launches it.
"""

import json
import os
import selectors
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import openai
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from rag import mcp_server
from rag.generator import NO_INFO_ANSWER, AnswerResult, Source, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "query_knowledge_base"

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


def offline_env() -> dict:
    """Environment without credentials, so any accidental DB or OpenAI use would fail loudly."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "POSTGRES_", "DATABASE_URL"))}
    env["PYTHONUNBUFFERED"] = "1"
    return env


class InMemoryTestCase(unittest.IsolatedAsyncioTestCase):
    async def call(self, arguments, result=ANSWER, error=None):
        with mock.patch.object(mcp_server, "answer", return_value=result, side_effect=error) as fake:
            async with Client(mcp_server.server) as client:
                return await client.call_tool(TOOL, arguments), fake


class ToolListingTest(InMemoryTestCase):
    async def test_exactly_one_tool_with_input_and_output_schema(self):
        async with Client(mcp_server.server) as client:
            tools = (await client.list_tools()).tools
        self.assertEqual([t.name for t in tools], [TOOL])
        tool = tools[0]
        self.assertTrue(tool.description)

        self.assertEqual(tool.input_schema["required"], ["question"])
        self.assertEqual(set(tool.input_schema["properties"]), {"question"})
        self.assertEqual(tool.input_schema["properties"]["question"]["type"], "string")

        out = tool.output_schema
        self.assertEqual(set(out["required"]), {"answer", "answerable", "sources"})
        source = out["$defs"]["KnowledgeSource"]
        self.assertEqual(set(source["required"]), {"label", "document_id", "chunk", "score", "cited"})
        score_doc = source["properties"]["score"]["description"]
        self.assertIn("1 - cosine_distance", score_doc)
        self.assertIn("Higher means more similar", score_doc)
        self.assertIn("not a probability", score_doc)

    async def test_listing_and_import_touch_neither_database_nor_openai(self):
        boom = AssertionError("must not be called")
        with mock.patch("rag.db.connect", side_effect=boom), \
             mock.patch("rag.embedder.make_client", side_effect=boom), \
             mock.patch("rag.generator.make_client", side_effect=boom):
            async with Client(mcp_server.server) as client:
                self.assertEqual(len((await client.list_tools()).tools), 1)


class ToolCallTest(InMemoryTestCase):
    async def test_answer_and_sources_are_returned_as_structured_and_text_content(self):
        result, fake = await self.call({"question": "  Can I get a refund for a monthly renewal charge?  "})
        fake.assert_called_once_with("Can I get a refund for a monthly renewal charge?", 3)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, {
            "answer": "Monthly renewals are not refundable. [S1]",
            "answerable": True,
            "sources": [
                {"label": "S1", "document_id": 3, "chunk": "Monthly renewals are not refundable.",
                 "score": 1 - 0.3775, "cited": True},
                {"label": "S2", "document_id": 3, "chunk": "Retries happen 3 times.",
                 "score": 1 - 0.4953, "cited": False},
            ],
        })
        # Same content for clients that only read text.
        self.assertEqual(len(result.content), 1)
        self.assertEqual(json.loads(result.content[0].text), result.structured_content)

    async def test_score_is_one_minus_cosine_distance(self):
        sources = [Source(f"S{i + 1}", 1, "T", "S", i, d, "x") for i, d in enumerate([0.0, 0.3775, 1.0, 1.2])]
        result, _ = await self.call({"question": "q"}, result=AnswerResult("q", True, "x [S1]", ["S1"], sources, "m", None))
        scores = [s["score"] for s in result.structured_content["sources"]]
        for got, want in zip(scores, [1.0, 0.6225, 0.0, -0.2]):
            self.assertAlmostEqual(got, want, places=9)
        self.assertEqual(scores, sorted(scores, reverse=True), "nearest first stays highest score first")

    async def test_unanswerable_result_is_not_an_error(self):
        no_info = AnswerResult("discount?", False, NO_INFO_ANSWER, [], [], model=None, usage=None)
        result, _ = await self.call({"question": "discount?"}, result=no_info)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, {"answer": NO_INFO_ANSWER, "answerable": False, "sources": []})

    async def test_invalid_input_is_a_tool_error_and_never_calls_answer(self):
        cases = [{"question": ""}, {"question": "   \n"}, {}, {"question": None}, {"question": 42}, {"question": ["a"]}]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result, fake = await self.call(arguments)
                self.assertTrue(result.is_error)
                self.assertIsNone(result.structured_content)
                self.assertTrue(result.content[0].text.startswith(f"Error executing tool {TOOL}"))
                fake.assert_not_called()

    async def test_blank_question_reports_the_validation_message(self):
        result, _ = await self.call({"question": "  "})
        self.assertEqual(result.content[0].text, f"Error executing tool {TOOL}: question must be a non-empty string")

    async def test_unexpected_errors_hide_internal_details(self):
        request = mock.Mock()
        errors = [
            RuntimeError("connection to postgres://rag:hunter2@localhost failed"),
            openai.APIConnectionError(message="proxy sk-proj-abc123 refused", request=request),
            KeyError("internal state"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with self.assertLogs(level="ERROR"):  # traceback goes to the server log only
                    result, _ = await self.call({"question": "refund?"}, error=error)
                self.assertTrue(result.is_error)
                text = result.content[0].text
                self.assertEqual(text, f"Error executing tool {TOOL}")
                for secret in ("hunter2", "sk-proj", "postgres://", "internal state"):
                    self.assertNotIn(secret, text)


class StdioSubprocessTest(unittest.TestCase):
    """The real transport: the server as a child process speaking JSON-RPC over stdin/stdout."""

    def start(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "rag.mcp_server"], cwd=REPO_ROOT, env=offline_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        def stop():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                pipe.close()

        self.addCleanup(stop)
        return proc

    def test_initialize_list_and_call_with_only_protocol_messages_on_stdout(self):
        proc = self.start()
        stdout_lines: list[str] = []
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        self.addCleanup(selector.close)

        def send(message):
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()

        def response(request_id, timeout=20.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not selector.select(timeout=0.5):
                    continue
                line = proc.stdout.readline()
                if not line:
                    self.fail(f"server closed stdout; stderr:\n{proc.stderr.read()}")
                stdout_lines.append(line)
                message = json.loads(line)  # anything that is not JSON-RPC fails the test here
                if message.get("id") == request_id:
                    return message
            self.fail(f"no response to request {request_id}")

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "stdio-test", "version": "0"}}})
        init = response(1)
        self.assertEqual(init["result"]["serverInfo"]["name"], "quillbox-knowledge-base")
        self.assertIn("tools", init["result"]["capabilities"])
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = response(2)["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], [TOOL])
        self.assertIn("outputSchema", tools[0])

        # A blank question fails validation before answer(), so no DB or OpenAI is needed.
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": TOOL, "arguments": {"question": " "}}})
        call = response(3)["result"]
        self.assertTrue(call["isError"])

        proc.stdin.close()
        proc.wait(timeout=10)
        stdout_lines.extend(proc.stdout.readlines())
        stderr = proc.stderr.read()
        for line in stdout_lines:
            self.assertEqual(json.loads(line)["jsonrpc"], "2.0", f"non-protocol stdout line: {line!r}")
        # The SDK logged the tool failure, and it went to stderr, not stdout.
        self.assertIn("query_knowledge_base", stderr)


class StdioSdkClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_client_lists_the_tool_over_stdio(self):
        params = StdioServerParameters(command=sys.executable, args=["-m", "rag.mcp_server"],
                                       cwd=str(REPO_ROOT), env=offline_env())
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
        self.assertEqual([t.name for t in tools], [TOOL])
        self.assertEqual(tools[0].input_schema["required"], ["question"])


if __name__ == "__main__":
    unittest.main()

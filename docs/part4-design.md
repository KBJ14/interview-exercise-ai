# Part 4 Design (MCP server)

Design and verification notes for the MCP server that exposes the RAG pipeline as a tool, as of commit `4f4759d`.
How to run and connect it: [run-guide.md, step 11](run-guide.md#11-mcp-server-part-4). Part 1 pipeline: [part1-design.md](part1-design.md). REST API: [part2-design.md](part2-design.md).

The server is named `quillbox-knowledge-base` after the sample corpus: Quillbox is a fictional SaaS product invented for this practice project (`data/documents/`), not a real service.

## Requirements

| README Part 4 requirement | Status | Code | Evidence |
|---------------------------|--------|------|----------|
| MCP tool `query_knowledge_base` | Done | `rag/mcp_server.py` (`@server.tool(name="query_knowledge_base")`) | `tests/test_mcp_server.py` (`test_exactly_one_tool_with_input_and_output_schema`) |
| Input `{ "question": string }` | Done | `query_knowledge_base(question: str)` | Input schema requires exactly `question` of type `string` |
| Output `{ "answer", "sources": [{ "document_id", "chunk", "score" }] }` | Done, with extra fields `answerable`, `label`, `cited` | `KnowledgeBaseAnswer`, `KnowledgeSource`, `to_tool_result()` | Output schema test; structured and text results test |
| Calls the RAG pipeline internally, no duplicated logic | Done | Calls only `rag.retriever.validate_query` and `rag.generator.answer()` | No retrieval, SQL, embedding or model call in `rag/mcp_server.py` |
| Clean separation of transport (MCP) and business logic (RAG) | Done | `rag/mcp_server.py` only validates, calls, and converts | Tests mock `answer()` and check only the MCP contract |
| Usable by an MCP-compatible client | Done for Claude Code | stdio entry point `python -m rag.mcp_server` | Real stdio subprocess tests; one live call from a Claude Code session (see [Verification](#verification)) |

Not covered: Claude Desktop, named in the README as an example client, was not tested; the live client was Claude Code.

## What the MCP tool is for

The REST API (Part 2) is for programs written against its documentation. The MCP tool is for LLM clients: the client asks the server which tools exist (`tools/list`), receives each tool's name, description and JSON Schemas, and the model decides when to call one (`tools/call`). Messages are JSON-RPC 2.0, here over stdio, and failures are reported either as protocol errors or as tool results marked `isError`.

The tool is an adapter around the existing pipeline, not a second implementation of it.

## Choices

| Choice | Reason |
|--------|--------|
| Python | The README allows Python or Go. The official Go SDK (`github.com/modelcontextprotocol/go-sdk`) requires Go 1.23 or newer in every release, and this environment has Go 1.22.2 |
| Official Python SDK `mcp==2.3.0` (`MCPServer`) | Supports Python 3.12 and fits the dependencies already in `.venv`; derives input and output schemas from type annotations; returns structured content with the same JSON as text; turns `ToolError` into `isError` results and hides other exceptions |
| stdio transport | The client starts the server as a subprocess, so nothing listens on a port. The MCP specification says clients should support stdio. Streamable HTTP would need Origin validation and authentication, which a local single-user setup does not need |
| Call `rag.generator.answer()` directly | Same retrieval and generation code as the CLI and FastAPI, with no Go or FastAPI process in between. The cost: the MCP process itself needs the database settings and `OPENAI_API_KEY` (see the run guide) |

Calling the Go `POST /query` instead would reuse Go's validation and error translation and keep the key out of the MCP process, but every tool call would then require both the Go and the FastAPI service to be running, and it would add an HTTP client and a second error mapping in Python.

## Request flow

```
MCP client (e.g. Claude Code)
   │  stdio: JSON-RPC  initialize → tools/list → tools/call
   ▼
rag/mcp_server.py  (transport layer)
   │  1. SDK validates arguments against the input schema
   │  2. validate_query(question, 3)        existing validation
   │  3. answer(question, 3)                existing pipeline
   │  4. to_tool_result(AnswerResult)       field mapping only
   ▼
rag.retriever.search  → OpenAI embeddings, PostgreSQL + pgvector
rag.generator         → OpenAI Responses API, citation checks
```

Importing the module or starting the server reaches neither PostgreSQL nor OpenAI; both are used only when the tool is called. `k` is fixed at the existing default of 3.

## Tool contract

Input: `{"question": "..."}`, a non-empty string after trimming (the same rule as the CLI and the REST API).

Output (`structuredContent`, also returned as the same JSON in one text content block):

| Field | Source in `AnswerResult` | Meaning |
|-------|--------------------------|---------|
| `answer` | `answer` | Answer grounded in the sources, citing them as `[S1]`, `[S2]`, … |
| `answerable` | `answerable` | `false` when the documents do not contain the answer; the answer then says so instead of guessing. This is a normal result, not an error |
| `sources` | `sources` | Every retrieved chunk, nearest first, whether cited or not |
| `sources[].label` | `Source.label` | The `[S#]` label the answer uses |
| `sources[].document_id` | `Source.document_id` | Document the chunk belongs to (`documents.id`) |
| `sources[].chunk` | `Source.body` | Chunk text without its title/section header line |
| `sources[].score` | `1 - Source.cosine_distance` | Cosine similarity between question and chunk embeddings, from -1 to 1; higher is more similar |
| `sources[].cited` | label in `citations` | Whether the answer actually cites this chunk |

`score` is a similarity measure, **not a probability or a confidence score**: it says how close two vectors are, not how likely the answer is to be correct. The REST API reports the same quantity as `cosine_distance` (smaller is closer); the MCP tool reports `score` because the README names that field.

Retrieved versus cited: `sources` lists everything retrieval returned (3 chunks). Only entries with `cited: true` are the evidence the answer relies on; the others were context the model chose not to use.

## Errors and stdio

| Situation | Result |
|-----------|--------|
| Missing or wrongly typed `question` | Rejected by the SDK's argument validation, returned as `isError` |
| Blank question | `ToolError` with the existing validation message: `Error executing tool query_knowledge_base: question must be a non-empty string` |
| Any other failure (database, OpenAI, unusable model output, a bug) | Left to the SDK, which returns only `Error executing tool query_knowledge_base`; the exception and traceback go to the server log |

- stdout carries MCP messages only. The module never prints, and the SDK's logging handlers write to stderr.
- Internal details (connection strings, keys, exception text) never reach the tool result.

## Limitations

- Failures after validation are returned as one generic tool error. The detailed categories the REST API distinguishes (`llm_unavailable`, `database_unavailable`, `llm_invalid_response`, …) are not exposed to the model yet; that needs the classification in `rag/api.py` moved into a shared module.
- The tool is a plain function. The SDK runs it in a worker thread, which keeps the protocol loop responsive, but this server does not pass cancellation into `answer()`: a cancelled request may leave its in-flight OpenAI or database call running until that call finishes or reaches its own timeout (30 seconds with 2 retries for OpenAI).
- `k` is fixed at 3; the tool has no parameter for it.
- Token usage is not part of the tool result and is not logged.
- The server serves one local client over stdio; there is no HTTP transport or authentication.
- Claude Desktop was not tested.

## Verification

### Automated tests (no database, no OpenAI)

`answer()` is mocked in every MCP test.

| Test | Covers |
|------|--------|
| `tests/test_mcp_server.py`, 8 in-memory tests (SDK `Client`) | One tool with its input and output schema, including the `score` description; listing touches neither database nor OpenAI; structured and text results agree; `score = 1 - cosine_distance`; unanswerable result is not an error; 6 invalid inputs never reach `answer()`; validation message; internal details hidden |
| `tests/test_mcp_server.py`, 2 stdio subprocess tests | `python -m rag.mcp_server` as a child process without credentials: raw JSON-RPC `initialize`, `tools/list` and a rejected `tools/call` with every stdout line a JSON-RPC message and the failure logged on stderr; the SDK's stdio client listing the tool |

All 10 MCP tests passed, and the full Python suite (79 tests) passed with none skipped.

### Live client check

These are separate from the automated tests:

1. **stdio process**: the server started with the command registered for Claude Code initialized and listed `query_knowledge_base` through the SDK's stdio client, without calling the tool.
2. **Claude Code**: after the local-scope registration (run guide, step 11), `claude mcp get` reported `✔ Connected`; the server process it started ran from the repository root with `OPENAI_API_KEY` present in its environment. The running session exposed the tool only after it reconnected to the server, as `mcp__quillbox-knowledge-base__query_knowledge_base`. It was called once, with real PostgreSQL and OpenAI:

| Item | Result |
|------|--------|
| Question | Can I get a refund for a monthly renewal charge? |
| `answerable` | `true` |
| Answer | Monthly renewal charges are not refundable; duplicate charges and billing errors caused by Quillbox are refunded in full regardless of when they are reported. Cites `[S1]` |
| `S1` | `billing.txt` (`document_id` 3), Refund Policy chunk, `cited: true`, `score` ≈ 0.62 |
| `S2`, `S3` | Other `billing.txt` chunks (failed payments, billing cycles), `cited: false` |

Both statements in the answer appear in the S1 chunk ("Monthly renewals are not refundable…", "Duplicate charges and billing errors caused by Quillbox are always refunded in full, regardless of when they are reported"). Token usage and cost of this call were not observed.

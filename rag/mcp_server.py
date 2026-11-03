"""MCP server exposing the RAG pipeline as one tool, query_knowledge_base, over stdio.

Run from the repo root:
    .venv/bin/python -m rag.mcp_server

This module is only the MCP transport layer: it validates the question with the existing
validate_query, calls the existing rag.generator.answer(), and converts its AnswerResult into
the tool's output schema. Retrieval and answer generation are not reimplemented here.

stdio rules: stdout carries MCP messages only. Nothing in this module prints; the SDK's logging
handlers write to stderr. Importing the module or starting the server touches neither the
database nor OpenAI; both are reached only when the tool is called.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rag.generator import AnswerResult, answer
from rag.retriever import DEFAULT_K, validate_query

server = MCPServer(
    name="quillbox-knowledge-base",
    instructions="Answers questions about the fictional Quillbox support documents, with cited sources.",
)


class KnowledgeSource(BaseModel):
    """One retrieved chunk, nearest first."""

    label: str = Field(description="Citation label used in the answer text, e.g. S1.")
    document_id: int = Field(description="Id of the document the chunk belongs to.")
    chunk: str = Field(description="Text of the retrieved chunk.")
    score: float = Field(
        description=(
            "Cosine similarity between the question and chunk embeddings, computed as "
            "1 - cosine_distance (range -1 to 1). Higher means more similar. It is a similarity "
            "measure, not a probability or a confidence score."
        )
    )
    cited: bool = Field(description="True if the answer cites this chunk.")


class KnowledgeBaseAnswer(BaseModel):
    """Grounded answer with the chunks it was generated from."""

    answer: str = Field(description="Answer grounded in the sources, with [S#] citations.")
    answerable: bool = Field(
        description="False when the documents do not contain the answer; the answer then says so."
    )
    sources: list[KnowledgeSource] = Field(description="Every retrieved chunk, nearest first.")


def to_tool_result(result: AnswerResult) -> KnowledgeBaseAnswer:
    return KnowledgeBaseAnswer(
        answer=result.answer,
        answerable=result.answerable,
        sources=[
            KnowledgeSource(
                label=s.label,
                document_id=s.document_id,
                chunk=s.body,
                score=1.0 - s.cosine_distance,
                cited=s.label in result.citations,
            )
            for s in result.sources
        ],
    )


# Plain `def`: answer() blocks on network I/O, and the SDK runs sync tools in a worker thread
# (anyio.to_thread.run_sync), so the protocol loop keeps serving while it runs.
@server.tool(
    name="query_knowledge_base",
    title="Query the knowledge base",
    description=(
        "Answer a natural-language question from the Quillbox support documents. "
        "Returns the answer, whether the documents contain it, and the retrieved chunks with "
        "similarity scores; the answer cites chunks by their [S#] labels."
    ),
    structured_output=True,  # fail at import if the return type ever stops producing a schema
)
def query_knowledge_base(
    question: Annotated[str, Field(description="The question to answer, in natural language.")],
) -> KnowledgeBaseAnswer:
    try:
        question = validate_query(question, DEFAULT_K)
    except ValueError as exc:
        # Anticipated input error: the message describes the caller's own input.
        raise ToolError(str(exc)) from exc
    # Any other exception (database, OpenAI, a bug) is left to the SDK, which returns only
    # "Error executing tool query_knowledge_base" and logs the traceback to stderr.
    return to_tool_result(answer(question, DEFAULT_K))


def main() -> None:
    server.run()  # stdio is the default transport


if __name__ == "__main__":
    main()

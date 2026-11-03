"""Internal HTTP service for the Go API: health check, question answering, ingestion.

Run from the repo root:
    .venv/bin/uvicorn rag.api:app --host 127.0.0.1 --port 8000

Every error response has the shape {"error": {"code": "...", "message": "..."}}.
Status codes separate who is at fault:
    400/413   the caller sent an invalid or oversized request
    502/503/504  an upstream dependency (OpenAI, PostgreSQL) failed or answered unusably
    500       a bug or misconfiguration in this service
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai
import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.chunker import ChunkingError
from rag.embedder import ConfigurationError, EmbeddingError
from rag.generator import AnswerResult, GenerationError, answer
from rag.ingest import FORMATS, IngestResult, ingest_text
from rag.loader import DocumentFormatError
from rag.retriever import DEFAULT_K, validate_query

# Largest accepted document, in UTF-8 bytes. All chunks are embedded in one API request, and
# 256 KiB (~65k tokens, ~220 chunks) stays well inside the embeddings request limits.
MAX_DOCUMENT_BYTES = 256 * 1024
# Whole request body: JSON escaping can grow text up to 6x (\u0001), plus the other fields.
MAX_INGEST_BODY_BYTES = 2 * 1024 * 1024
MAX_SOURCE_CHARS = 255
MAX_TITLE_CHARS = 300

logger = logging.getLogger(__name__)

app = FastAPI(title="Quillbox RAG (internal)", docs_url=None, redoc_url=None)


class QueryRequest(BaseModel):
    # Fields are untyped on purpose: type and range checks live in validate_query,
    # so the HTTP service and the CLI reject exactly the same inputs.
    model_config = ConfigDict(extra="forbid")
    question: Any = None
    k: Any = DEFAULT_K


class IngestRequest(BaseModel):
    # Untyped for the same reason as QueryRequest: _validate_ingest gives precise messages.
    model_config = ConfigDict(extra="forbid")
    text: Any = None
    source: Any = None  # file name or label; metadata only, never opened as a path
    title: Any = None
    format: Any = "plain"


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ServiceError)
async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
    return error_response(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default is 422 with its own body; keep one error shape for the Go client.
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    return error_response(400, "invalid_request", f"{first.get('msg', 'invalid request')}" + (f" ({where})" if where else ""))


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    codes = {404: "not_found", 405: "method_not_allowed"}
    return error_response(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(Exception)
async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    # Details go to the server log only; the response must not leak internals.
    logger.exception("unhandled error", exc_info=exc)
    return error_response(500, "internal_error", "unexpected internal error")


@app.get("/health")
def health() -> dict:
    # Liveness only: deliberately touches neither the database nor OpenAI.
    return {"status": "ok"}


# Plain `def`: answer() blocks on network I/O, so FastAPI runs it in a worker thread.
@app.post("/internal/query")
def query(req: QueryRequest) -> dict:
    try:
        question = validate_query(req.question, req.k)
    except ValueError as exc:
        raise ServiceError(400, "invalid_request", str(exc)) from exc

    try:
        result = answer(question, req.k)
    except Exception as exc:
        raise _classify(exc) from exc
    return _to_json(result)


# async so the body can be read with a size cap before anything is parsed; the blocking
# ingest work (DB, OpenAI) then runs in a worker thread like /internal/query does.
@app.post("/internal/ingest")
async def ingest(request: Request) -> JSONResponse:
    body = await _read_body(request, MAX_INGEST_BODY_BYTES)
    try:
        req = IngestRequest.model_validate(json.loads(body))
    except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
        raise ServiceError(400, "invalid_request", _first_error(exc)) from exc
    text, source, title, fmt = _validate_ingest(req)

    try:
        result = await run_in_threadpool(ingest_text, text, source=source, title=title, fmt=fmt)
    except (DocumentFormatError, ChunkingError) as exc:
        # The text itself cannot be used (wrong layout, empty, a word longer than a chunk).
        raise ServiceError(400, "invalid_document", str(exc)) from exc
    except Exception as exc:
        raise _classify(exc) from exc
    status = 201 if result.status == "inserted" else 200
    return JSONResponse(status_code=status, content=_ingest_json(result))


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise ServiceError(413, "request_too_large", f"request body must be at most {limit} bytes")
    chunks, size = [], 0
    async for chunk in request.stream():  # also covers chunked bodies without Content-Length
        size += len(chunk)
        if size > limit:
            raise ServiceError(413, "request_too_large", f"request body must be at most {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return f"{first['msg']} ({'.'.join(str(p) for p in first['loc'])})"
    return "request body must be a JSON object"


def _validate_ingest(req: IngestRequest) -> tuple[str, str | None, str | None, str]:
    if not isinstance(req.text, str) or not req.text.strip():
        raise ServiceError(400, "invalid_request", "text must be a non-empty string")
    if len(req.text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ServiceError(413, "document_too_large", f"document must be at most {MAX_DOCUMENT_BYTES} bytes of UTF-8 text")
    for name, value, limit in (("source", req.source, MAX_SOURCE_CHARS), ("title", req.title, MAX_TITLE_CHARS)):
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > limit):
            raise ServiceError(400, "invalid_request", f"{name} must be a non-empty string of at most {limit} characters")
    if req.format not in FORMATS:
        raise ServiceError(400, "invalid_request", f"format must be one of {', '.join(FORMATS)}")
    return req.text, req.source, req.title, req.format


def _ingest_json(result: IngestResult) -> dict:
    return {"status": result.status, "document_id": result.document_id, "chunk_count": result.chunk_count}


def _classify(exc: Exception) -> Exception:
    """Map a failure from answer() to an HTTP error by what actually went wrong."""
    if isinstance(exc, ConfigurationError):
        logger.error("configuration error: %s", exc)
        return ServiceError(500, "configuration_error", "service is not configured for the language model")
    if isinstance(exc, GenerationError):
        # The model responded, but the output was incomplete or failed format/citation checks.
        return ServiceError(502, "llm_invalid_response", f"language model returned an unusable answer: {exc}")
    if isinstance(exc, EmbeddingError):
        return ServiceError(502, "embedding_invalid_response", f"embedding service returned an unexpected result: {exc}")
    if isinstance(exc, openai.APITimeoutError):  # subclass of APIConnectionError, so check first
        return ServiceError(504, "llm_timeout", "OpenAI API did not respond in time")
    if isinstance(exc, openai.APIConnectionError):
        return ServiceError(503, "llm_unavailable", "could not reach OpenAI API")
    if isinstance(exc, openai.RateLimitError):
        return ServiceError(503, "llm_rate_limited", "OpenAI API rate limit or quota reached")
    if isinstance(exc, openai.APIStatusError):
        # Bodies are not echoed: a 401 body can contain a masked key fragment.
        if exc.status_code >= 500:
            return ServiceError(503, "llm_unavailable", f"OpenAI API error (HTTP {exc.status_code})")
        logger.error("OpenAI rejected the request: HTTP %s %s", exc.status_code, type(exc).__name__)
        return ServiceError(502, "llm_request_rejected", f"OpenAI API rejected the request (HTTP {exc.status_code})")
    if isinstance(exc, psycopg.OperationalError):
        return ServiceError(503, "database_unavailable", "database is unreachable")
    if isinstance(exc, ValueError):
        # Input was validated above, so a ValueError here is a bug, not a caller mistake.
        logger.exception("unexpected ValueError after validation", exc_info=exc)
        return ServiceError(500, "internal_error", "unexpected internal error")
    logger.exception("unhandled error in answer()", exc_info=exc)
    return ServiceError(500, "internal_error", "unexpected internal error")


def _to_json(result: AnswerResult) -> dict:
    return {
        "question": result.question,
        "answerable": result.answerable,
        "answer": result.answer,
        "citations": result.citations,
        "sources": [
            {
                "label": s.label,
                "document_id": s.document_id,
                "title": s.title,
                "section": s.section,
                "chunk_index": s.chunk_index,
                "chunk": s.body,
                # Distance, not a probability: 0 is the same direction, smaller is closer.
                "cosine_distance": s.cosine_distance,
                "cited": s.label in result.citations,
            }
            for s in result.sources
        ],
        "model": result.model,
        "usage": None if result.usage is None else {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }

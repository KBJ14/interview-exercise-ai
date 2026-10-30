"""Create embeddings with the OpenAI API.

Premise (same as schema.sql): a single model, text-embedding-3-small at 1536 dimensions.
The API key is read from the OPENAI_API_KEY environment variable and never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # must match vector(1536) in schema.sql

# Bounded retries: the SDK retries connection errors, 429 and 5xx with backoff, then raises.
MAX_RETRIES = 2
TIMEOUT_SECONDS = 30.0


class EmbeddingError(RuntimeError):
    """The API answered, but not with what the schema expects."""


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    vectors: list[list[float]]
    prompt_tokens: int
    total_tokens: int


def make_client() -> OpenAI:
    # Checked here so a missing key fails with a clear message instead of an SDK traceback.
    if not os.environ.get("OPENAI_API_KEY"):
        raise EmbeddingError("OPENAI_API_KEY is not set (expected from ~/.config/openai/env)")
    return OpenAI(max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS)


def embed_texts(texts: list[str], client: OpenAI | None = None) -> EmbeddingResult:
    if not texts:
        raise ValueError("texts must not be empty")

    client = client or make_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        # 1536 is already the model's default; passing it pins the size the schema depends on.
        dimensions=EMBEDDING_DIMENSIONS,
    )

    # The API may return items out of order; `index` maps each vector back to its input.
    vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    if len(vectors) != len(texts):
        raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(vectors)}")
    bad = {len(v) for v in vectors} - {EMBEDDING_DIMENSIONS}
    if bad:
        raise EmbeddingError(f"expected {EMBEDDING_DIMENSIONS} dimensions, got {sorted(bad)}")

    return EmbeddingResult(
        model=response.model,
        vectors=vectors,
        prompt_tokens=response.usage.prompt_tokens,
        total_tokens=response.usage.total_tokens,
    )

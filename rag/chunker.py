"""Split parsed documents into token-bounded chunks for embedding.

Each section becomes one chunk when it fits. A larger section is packed sentence by sentence,
and consecutive chunks of the same section share up to `overlap_tokens` of trailing sentences.
Every chunk starts with a "<title> > <heading>" line so the topic is part of what gets embedded,
and `max_tokens` is checked against that full text (header + overlap + body).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from rag.loader import Document

DEFAULT_MAX_TOKENS = 300
DEFAULT_OVERLAP_TOKENS = 50

# cl100k_base is the tokenizer of text-embedding-3-small, so counts match what the API sees.
ENCODING_NAME = "cl100k_base"

# Good enough for the practice corpus; abbreviations like "e.g." would be split wrongly.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

TokenCounter = Callable[[str], int]


class ChunkingError(ValueError):
    """Text cannot be fitted into max_tokens without dropping content."""


@dataclass(frozen=True)
class Chunk:
    source_file: str
    chunk_index: int  # 0-based position within the document
    section: str
    content: str  # header line + body: exactly the text to embed
    token_count: int


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # First use downloads the encoding file once; later runs use tiktoken's local cache.
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_document(
    doc: Document,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count: TokenCounter = count_tokens,
) -> list[Chunk]:
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(f"overlap_tokens must be in [0, max_tokens), got {overlap_tokens}")

    chunks: list[Chunk] = []
    for section in doc.sections:
        header = f"{doc.title} > {section.heading}"
        for body in _split_section(header, section.body, max_tokens, overlap_tokens, count):
            content = _render(header, [body])
            chunks.append(Chunk(doc.source, len(chunks), section.heading, content, count(content)))
    return chunks


def _render(header: str, units: list[str]) -> str:
    return f"{header}\n{' '.join(units)}"


def _split_section(
    header: str, body: str, max_tokens: int, overlap_tokens: int, count: TokenCounter
) -> list[str]:
    def fits(units: list[str]) -> bool:
        # Count the rendered text, not a sum of parts: BPE token counts are not additive.
        return count(_render(header, units)) <= max_tokens

    if not fits([]):
        raise ChunkingError(f"heading line alone exceeds {max_tokens} tokens: {header!r}")
    if fits([body]):
        return [body]

    units: list[str] = []
    for sentence in _SENTENCE_END.split(body):
        units.extend([sentence] if fits([sentence]) else _split_long_sentence(sentence, fits))

    bodies: list[str] = []
    current: list[str] = []
    for unit in units:
        if fits(current + [unit]):
            current.append(unit)
            continue
        bodies.append(" ".join(current))
        overlap = _tail_within(current, overlap_tokens, count)
        # The size cap wins over the overlap target: drop the oldest overlap sentences until it fits.
        while overlap and not fits(overlap + [unit]):
            overlap = overlap[1:]
        current = overlap + [unit]
    bodies.append(" ".join(current))
    return bodies


def _split_long_sentence(sentence: str, fits: Callable[[list[str]], bool]) -> list[str]:
    """Split a sentence that alone exceeds the limit at word boundaries.

    The pieces keep every word in order, so nothing is truncated; only the sentence boundary
    is lost. A single word that cannot fit is an error rather than a silent cut.
    """
    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if fits(current + [word]):
            current.append(word)
            continue
        if not current:
            raise ChunkingError(f"a single word exceeds the chunk limit: {word[:40]!r}...")
        pieces.append(" ".join(current))
        current = [word]
        if not fits(current):
            raise ChunkingError(f"a single word exceeds the chunk limit: {word[:40]!r}...")
    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail_within(units: list[str], budget: int, count: TokenCounter) -> list[str]:
    """Longest run of trailing units whose joined text is within `budget` tokens (may be empty)."""
    tail: list[str] = []
    for unit in reversed(units):
        candidate = [unit] + tail
        if count(" ".join(candidate)) > budget:
            break
        tail = candidate
    return tail

"""Load the practice corpus from plain-text files.

Limitation: this parser only understands the layout of the fictional Quillbox documents
in data/documents/. Blocks are separated by blank lines:
an optional "[FICTIONAL ...]" notice, a one-line title, then sections whose first line
is the heading and whose remaining lines are the body.
Any other layout raises DocumentFormatError instead of being guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NOTICE_PREFIX = "[FICTIONAL"
_BLOCK_SEPARATOR = re.compile(r"\n\s*\n")


class DocumentFormatError(ValueError):
    """The file does not follow the expected title/section layout."""


@dataclass(frozen=True)
class Section:
    heading: str
    body: str


@dataclass(frozen=True)
class Document:
    source: str  # file name, e.g. "billing.txt"
    title: str
    raw_text: str  # untouched file content (notice included), kept for re-chunking later
    sections: tuple[Section, ...]


def load_documents(directory: str | Path) -> list[Document]:
    """Parse every *.txt file directly inside `directory`, sorted by file name.

    A relative `directory` resolves against the current working directory.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"document directory not found: {root}")

    # Non-recursive and .txt only, so answer keys such as docs/rag-checks.md never enter the corpus.
    paths = sorted(p for p in root.glob("*.txt") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no .txt documents in {root}")

    return [parse_document(p.name, p.read_text(encoding="utf-8")) for p in paths]


def parse_document(source: str, text: str) -> Document:
    blocks = [b.strip() for b in _BLOCK_SEPARATOR.split(text.strip()) if b.strip()]

    # The notice is identical in every file; keeping it in chunks would make all documents
    # look similar to the retriever. It stays in raw_text.
    if blocks and blocks[0].startswith(NOTICE_PREFIX):
        blocks = blocks[1:]

    if len(blocks) < 2:
        raise DocumentFormatError(f"{source}: expected a title block followed by at least one section")

    title, *section_blocks = blocks
    if "\n" in title:
        raise DocumentFormatError(f"{source}: title block must be a single line, got {title!r}")

    sections = []
    for block in section_blocks:
        heading, _, body = block.partition("\n")
        # Collapse line breaks and repeated spaces so chunk text can be rebuilt word for word.
        body = " ".join(body.split())
        if not body:
            raise DocumentFormatError(f"{source}: section {heading.strip()!r} has no body")
        sections.append(Section(heading.strip(), body))

    return Document(source=source, title=title, raw_text=text, sections=tuple(sections))

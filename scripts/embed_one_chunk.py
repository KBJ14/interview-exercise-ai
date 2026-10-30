"""Embed a single chunk with the OpenAI API and print a summary (never the vector or the key).

Usage:
    python scripts/embed_one_chunk.py [--file billing.txt] [--section "Refund Policy"]

Makes exactly one embeddings request. Nothing is written to the database.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Lets the script run by path from any working directory without installing the package.
sys.path.insert(0, str(REPO_ROOT))

import openai  # noqa: E402

from rag.chunker import chunk_document  # noqa: E402
from rag.embedder import EmbeddingError, embed_texts  # noqa: E402
from rag.loader import load_documents  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "documents")
    parser.add_argument("--file", default="billing.txt")
    parser.add_argument("--section", default="Refund Policy")
    args = parser.parse_args(argv)

    docs = {d.source: d for d in load_documents(args.dir)}
    if args.file not in docs:
        print(f"error: {args.file} not found in {args.dir}", file=sys.stderr)
        return 1
    matches = [c for c in chunk_document(docs[args.file]) if c.section == args.section]
    if len(matches) != 1:
        print(f"error: expected exactly 1 chunk for section {args.section!r}, found {len(matches)}", file=sys.stderr)
        return 1
    chunk = matches[0]

    try:
        result = embed_texts([chunk.content])
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except openai.APIStatusError as exc:
        # Status and error type only; the SDK message can echo a masked key fragment.
        print(f"error: OpenAI API returned HTTP {exc.status_code} ({type(exc).__name__})", file=sys.stderr)
        return 1
    except openai.APIConnectionError as exc:
        print(f"error: could not reach OpenAI API ({type(exc).__name__})", file=sys.stderr)
        return 1

    print(f"chunk:            {chunk.source_file} [{chunk.chunk_index}] {chunk.section}")
    print(f"model:            {result.model}")
    print(f"chunk tokens:     {chunk.token_count} (tiktoken, local)")
    print(f"vector dims:      {len(result.vectors[0])}")
    print(f"usage:            prompt_tokens={result.prompt_tokens} total_tokens={result.total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
